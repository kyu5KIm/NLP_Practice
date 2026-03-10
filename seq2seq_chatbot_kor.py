"""
한국어 Seq2Seq Q&A Chatbot (PyTorch)

설명
- 데이터셋: songys/Chatbot_data (한국어 문답 페어)
- 모델: GRU 기반 Encoder-Decoder Seq2Seq
- 전처리: 문장 정규화 + 토큰화 + Vocabulary 구축
- 학습: Teacher Forcing + PAD 무시 손실 계산
- 추론: Greedy Decoding + while문 챗봇

주의
- 업로드해 주신 ko_stopwords.txt 는 불용어가 많이 포함되어 있지만,
  생성형 챗봇(Seq2Seq)에서는 조사를 포함한 기능어를 지우면 문장 품질이 크게 떨어질 수 있어
  기본값은 use_stopwords=False 로 두었습니다.
- 즉, 이 스크립트는 "불용어 파일을 읽을 수는 있지만 기본적으로는 사용하지 않음"이 핵심입니다.
"""

from __future__ import annotations

import math
import os
import random
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. 실행 환경/하이퍼파라미터 설정
# ============================================================

@dataclass
class Config:
    # ------------------------------
    # 데이터/파일 경로
    # ------------------------------
    dataset_url: str = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"
    dataset_path: str = "ChatbotData.csv"
    stopwords_path: str = "ko_stopwords.txt"
    checkpoint_path: str = "seq2seq_chatbot_best.pt"

    # ------------------------------
    # 전처리 옵션
    # ------------------------------
    use_stopwords: bool = False              # 생성형 챗봇에서는 기본적으로 False 권장
    remove_duplicate_pairs: bool = True      # 완전히 같은 Q-A 쌍은 제거
    min_freq: int = 2                        # 너무 드문 토큰은 <unk> 처리
    max_vocab_size: int | None = None        # None 이면 제한 없음
    max_src_len: int = 20                    # 질문 최대 토큰 길이(EOS 포함 전 기준)
    max_trg_len: int = 26                    # 답변 최대 토큰 길이(SOS/EOS 포함 전 기준)

    # ------------------------------
    # 데이터 분할/로더 옵션
    # ------------------------------
    valid_size: float = 0.1
    batch_size: int = 128
    num_workers: int = 0

    # ------------------------------
    # 모델 옵션
    # ------------------------------
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True

    # ------------------------------
    # 학습 옵션
    # ------------------------------
    epochs: int = 10
    lr: float = 1e-3
    teacher_forcing_ratio: float = 0.5
    grad_clip: float = 1.0
    seed: int = 42

    # ------------------------------
    # 추론 옵션
    # ------------------------------
    max_decode_len: int = 30

    # ------------------------------
    # 기타
    # ------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_num_threads: int = 1              # CPU 학습 시 과도한 스레드 오버헤드를 줄이기 위한 설정


# 스크립트가 위치한 폴더를 기준 디렉터리로 사용한다.
BASE_DIR = Path(__file__).resolve().parent
CONFIG = Config(
    dataset_path=str(BASE_DIR / "ChatbotData.csv"),
    stopwords_path=str(BASE_DIR / "ko_stopwords.txt"),
    checkpoint_path=str(BASE_DIR / "seq2seq_chatbot_best.pt"),
)


# ============================================================
# 2. 재현 가능한 결과를 위한 시드 고정
# ============================================================

def set_seed(seed: int) -> None:
    """파이썬/파이토치 시드를 고정해 결과 재현성을 높인다."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_threads(num_threads: int) -> None:
    """
    CPU 환경에서 스레드 수를 제한한다.

    이유
    - 일부 환경에서는 GRU/RNN이 스레드를 너무 많이 사용하면서
      오히려 학습/추론이 극단적으로 느려질 수 있다.
    - 이 스크립트는 기본값을 1로 두어 과도한 스레드 오버헤드를 피한다.
    """
    if num_threads < 1:
        num_threads = 1

    torch.set_num_threads(num_threads)

    # set_num_interop_threads는 이미 병렬 연산이 시작된 뒤에는 RuntimeError가 날 수 있으므로 예외 처리한다.
    try:
        torch.set_num_interop_threads(min(num_threads, 2))
    except RuntimeError:
        pass


# ============================================================
# 3. 불용어 파일 로딩
# ============================================================

def load_stopwords(stopwords_path: str) -> Tuple[set[str], set[str]]:
    """
    불용어 파일을 읽어 단일 토큰 불용어와 다중 어절 불용어를 분리한다.

    왜 분리하나?
    - '을', '를', '은' 처럼 토큰 1개짜리 불용어는 토큰 단위 제거가 가능하다.
    - '예를 들면', '할 수 있다' 같은 다중 어절 불용어는
      현재의 간단한 토큰화 방식에서는 안전하게 제거하기 어렵다.
    - 게다가 생성형 챗봇에서는 이런 제거 자체가 문장 품질을 해칠 가능성이 높다.
    """
    path = Path(stopwords_path)
    if not path.exists():
        return set(), set()

    seen = set()
    cleaned_words: List[str] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        # 탭이 섞인 줄이 일부 있어서 탭 단위로 먼저 분리한다.
        for piece in raw_line.split("\t"):
            piece = re.sub(r"\s+", " ", piece.strip())
            if piece and piece not in seen:
                cleaned_words.append(piece)
                seen.add(piece)

    single_token_stopwords = {w for w in cleaned_words if " " not in w}
    multi_token_stopwords = {w for w in cleaned_words if " " in w}
    return single_token_stopwords, multi_token_stopwords


# ============================================================
# 4. 텍스트 전처리
# ============================================================

# 한국어/영문/숫자는 하나의 토큰으로 묶고,
# 나머지 특수문자/이모티콘/구두점은 1글자씩 토큰으로 분리한다.
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+|[^\s]")


def normalize_text(text: str) -> str:
    """
    문장을 모델에 넣기 전 최소한으로 정리한다.

    여기서 강한 정규화를 하지 않는 이유
    - 챗봇은 문장 생성이 목적이라 원문 느낌을 어느 정도 살리는 편이 낫다.
    - 너무 과한 정규화/불용어 제거는 응답 품질을 떨어뜨릴 수 있다.
    """
    text = str(text).strip()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str, use_stopwords: bool = False, stopwords: set[str] | None = None) -> List[str]:
    """
    문장을 토큰 리스트로 변환한다.

    use_stopwords=False 를 기본값으로 둔 이유
    - Seq2Seq 생성 모델은 조사/어미/기능어가 중요하다.
    - 불용어를 무턱대고 지우면 '문맥은 남고 문장성은 무너지는' 경우가 많다.
    """
    text = normalize_text(text)
    tokens = TOKEN_PATTERN.findall(text)

    if use_stopwords and stopwords:
        tokens = [tok for tok in tokens if tok not in stopwords]

    return tokens


def detokenize(tokens: Sequence[str]) -> str:
    """
    토큰 리스트를 사람이 읽기 쉬운 문장으로 복원한다.
    간단한 규칙 기반 후처리만 적용한다.
    """
    sentence = " ".join(tokens)
    # 구두점 앞 공백 제거
    sentence = re.sub(r"\s+([?.!,~])", r"\1", sentence)
    # 괄호 주변 공백 정리
    sentence = re.sub(r"\(\s+", "(", sentence)
    sentence = re.sub(r"\s+\)", ")", sentence)
    # 여러 공백 하나로 축소
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


# ============================================================
# 5. 데이터셋 다운로드 및 로딩
# ============================================================

def download_dataset_if_needed(dataset_url: str, dataset_path: str) -> None:
    """데이터셋 파일이 없으면 다운로드한다."""
    path = Path(dataset_path)
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 데이터셋 다운로드: {dataset_url}")
    urllib.request.urlretrieve(dataset_url, path)
    print(f"[INFO] 저장 완료: {path}")


def load_dataframe(config: Config, single_stopwords: set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    CSV를 읽고 전처리한 뒤 train/valid 데이터프레임으로 나눈다.
    label 컬럼을 이용해 stratified split(비율 유지 분할)을 수행한다.
    """
    df = pd.read_csv(config.dataset_path)

    # 기본 컬럼명 검증
    required_cols = {"Q", "A", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 컬럼이 예상과 다릅니다. 필요 컬럼: {required_cols}, 실제 컬럼: {df.columns.tolist()}")

    # 결측 제거
    df = df.dropna(subset=["Q", "A"]).copy()

    # 텍스트 정규화
    df["Q"] = df["Q"].apply(normalize_text)
    df["A"] = df["A"].apply(normalize_text)

    # 완전히 같은 Q-A 쌍 제거 (선택)
    if config.remove_duplicate_pairs:
        df = df.drop_duplicates(subset=["Q", "A"]).reset_index(drop=True)

    # 토큰 길이 계산
    df["q_tokens"] = df["Q"].apply(lambda x: tokenize(x, config.use_stopwords, single_stopwords))
    df["a_tokens"] = df["A"].apply(lambda x: tokenize(x, config.use_stopwords, single_stopwords))
    df["q_len"] = df["q_tokens"].apply(len)
    df["a_len"] = df["a_tokens"].apply(len)

    # 길이 제한을 넘는 샘플 제거
    # - 질문은 EOS를 뒤에 붙일 예정이므로 max_src_len - 1 까지만 허용
    # - 답변은 SOS/EOS를 붙일 예정이므로 max_trg_len - 2 까지만 허용
    df = df[
        (df["q_len"] <= config.max_src_len - 1) &
        (df["a_len"] <= config.max_trg_len - 2)
    ].reset_index(drop=True)

    # label 분포를 유지하면서 train/valid 분할
    train_df, valid_df = train_test_split(
        df,
        test_size=config.valid_size,
        random_state=config.seed,
        stratify=df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    return train_df, valid_df


# ============================================================
# 6. Vocabulary 구축
# ============================================================

class Vocabulary:
    """
    토큰 <-> 인덱스 매핑을 담당하는 간단한 Vocabulary 클래스
    """

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq: int = 1, max_size: int | None = None) -> None:
        self.min_freq = min_freq
        self.max_size = max_size

        self.special_tokens = [
            self.PAD_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
        ]

        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def build(self, tokenized_texts: Iterable[Sequence[str]]) -> None:
        """학습 데이터 토큰 빈도를 세어 Vocabulary를 만든다."""
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        # 최소 빈도 기준 적용
        items = [(tok, freq) for tok, freq in counter.items() if freq >= self.min_freq]

        # 자주 나온 토큰부터 정렬
        items.sort(key=lambda x: (-x[1], x[0]))

        # 최대 vocab 크기 제한이 있으면 적용
        if self.max_size is not None:
            keep_n = max(self.max_size - len(self.special_tokens), 0)
            items = items[:keep_n]

        # special token 먼저 등록
        all_tokens = self.special_tokens + [tok for tok, _ in items]

        self.stoi = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.stoi[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[self.EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[self.UNK_TOKEN]

    def numericalize(self, tokens: Sequence[str]) -> List[int]:
        """토큰 리스트를 인덱스 리스트로 변환한다."""
        return [self.stoi.get(tok, self.unk_idx) for tok in tokens]

    def denumericalize(self, ids: Sequence[int]) -> List[str]:
        """인덱스 리스트를 토큰 리스트로 변환한다."""
        return [self.itos.get(idx, self.UNK_TOKEN) for idx in ids]


# ============================================================
# 7. Dataset / DataLoader
# ============================================================

class ChatDataset(Dataset):
    """
    질문/답변 텍스트를 실제 학습용 텐서로 바꿔 주는 Dataset
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocab: Vocabulary,
        config: Config,
        single_stopwords: set[str],
    ) -> None:
        self.df = dataframe
        self.vocab = vocab
        self.config = config
        self.single_stopwords = single_stopwords

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # 질문 토큰화
        src_tokens = tokenize(
            row["Q"],
            use_stopwords=self.config.use_stopwords,
            stopwords=self.single_stopwords,
        )
        src_tokens = src_tokens[: self.config.max_src_len - 1]
        src_tokens = src_tokens + [self.vocab.EOS_TOKEN]

        # 답변 토큰화
        trg_tokens = tokenize(
            row["A"],
            use_stopwords=self.config.use_stopwords,
            stopwords=self.single_stopwords,
        )
        trg_tokens = trg_tokens[: self.config.max_trg_len - 2]
        trg_tokens = [self.vocab.SOS_TOKEN] + trg_tokens + [self.vocab.EOS_TOKEN]

        src_ids = self.vocab.numericalize(src_tokens)
        trg_ids = self.vocab.numericalize(trg_tokens)

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def build_collate_fn(pad_idx: int):
    """
    가변 길이 시퀀스를 배치 단위로 PAD 해주는 collate_fn 생성기

    왜 필요하나?
    - 질문/답변 길이가 서로 다르기 때문에 기본 collate로는 배치 텐서로 묶기 어렵다.
    - pad_sequence로 가장 긴 문장 길이에 맞춰 패딩해야 한다.
    """
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        src_batch, trg_batch = zip(*batch)

        src_lengths = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
        trg_lengths = torch.tensor([len(x) for x in trg_batch], dtype=torch.long)

        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=pad_idx)

        return src_padded, src_lengths, trg_padded, trg_lengths

    return collate_fn


def build_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    vocab: Vocabulary,
    config: Config,
    single_stopwords: set[str],
) -> Tuple[DataLoader, DataLoader]:
    """
    Dataset과 DataLoader를 생성한다.
    """
    train_dataset = ChatDataset(train_df, vocab, config, single_stopwords)
    valid_dataset = ChatDataset(valid_df, vocab, config, single_stopwords)

    collate_fn = build_collate_fn(vocab.pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader


# ============================================================
# 8. Encoder / Decoder / Seq2Seq 모델
# ============================================================

class Encoder(nn.Module):
    """
    질문 문장을 읽어 hidden state로 압축하는 인코더
    - Embedding
    - GRU
    - (선택) 양방향 인코더 hidden을 디코더 hidden 차원으로 투영
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # 양방향일 때는 forward/backward hidden을 이어 붙인 뒤 다시 hidden_dim으로 줄여 준다.
        if bidirectional:
            self.hidden_bridge = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.hidden_bridge = None

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src: [batch_size, src_len]
        src_lengths: [batch_size]

        반환
        - outputs: 각 시점의 encoder output
        - hidden: decoder 초기 hidden으로 사용할 최종 hidden
        """
        embedded = self.dropout(self.embedding(src))

        # pack_padded_sequence:
        # PAD 토큰을 제외하고 실제 길이만 RNN에 넣기 위해 사용한다.
        packed = pack_padded_sequence(
            embedded,
            lengths=src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_outputs, hidden = self.gru(packed)

        # packed된 출력은 다시 padded tensor로 풀지 않아도
        # 현재 기본 Seq2Seq에서는 hidden만 있으면 되지만,
        # 확장성을 위해 일단 풀어 둔다.
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            padding_value=0.0,
        )

        if self.bidirectional:
            batch_size = src.size(0)

            # hidden: [num_layers * 2, batch_size, hidden_dim]
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)

            # 각 층마다 forward/backward hidden을 이어 붙인다.
            # 결과 shape: [num_layers, batch_size, hidden_dim * 2]
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)

            # 디코더 hidden 크기에 맞춰 투영
            hidden = torch.tanh(self.hidden_bridge(hidden))

        return outputs, hidden


class Decoder(nn.Module):
    """
    이전에 생성한 토큰 1개와 hidden state를 받아
    다음 토큰 분포를 예측하는 디코더
    """

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_token: [batch_size]
        hidden: [num_layers, batch_size, hidden_dim]

        반환
        - logits: [batch_size, vocab_size]
        - hidden: 다음 시점 hidden
        """
        input_token = input_token.unsqueeze(1)              # [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token)) # [batch_size, 1, embedding_dim]

        output, hidden = self.gru(embedded, hidden)         # output: [batch_size, 1, hidden_dim]
        logits = self.fc_out(output.squeeze(1))             # [batch_size, vocab_size]

        return logits, hidden


class Seq2Seq(nn.Module):
    """
    Encoder + Decoder 를 묶은 전체 Seq2Seq 모델
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        src: [batch_size, src_len]
        trg: [batch_size, trg_len]

        반환
        - outputs: [batch_size, trg_len - 1, vocab_size]
          (trg[:, 1:] 를 예측한 결과)
        """
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len - 1, vocab_size, device=self.device)

        _, hidden = self.encoder(src, src_lengths)

        # 첫 입력은 항상 <sos>
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t - 1] = logits

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)

            # teacher forcing이면 정답 토큰을 넣고,
            # 아니면 모델이 방금 예측한 토큰을 다음 입력으로 넣는다.
            input_token = trg[:, t] if teacher_force else top1

        return outputs


# ============================================================
# 9. 학습/평가 함수
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """학습 가능한 파라미터 수를 계산한다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: Seq2Seq,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: Config,
) -> float:
    """한 epoch 동안 학습을 수행하고 평균 loss를 반환한다."""
    model.train()
    epoch_loss = 0.0

    for src, src_lengths, trg, _ in dataloader:
        src = src.to(config.device)
        src_lengths = src_lengths.to(config.device)
        trg = trg.to(config.device)

        optimizer.zero_grad()

        outputs = model(
            src=src,
            src_lengths=src_lengths,
            trg=trg,
            teacher_forcing_ratio=config.teacher_forcing_ratio,
        )

        # outputs: [batch_size, trg_len - 1, vocab_size]
        # target : [batch_size, trg_len - 1]
        output_dim = outputs.size(-1)

        loss = criterion(
            outputs.reshape(-1, output_dim),
            trg[:, 1:].reshape(-1),
        )

        loss.backward()

        # gradient explosion 방지
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: Seq2Seq,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: Config,
) -> float:
    """검증 데이터셋에 대한 평균 loss를 계산한다."""
    model.eval()
    epoch_loss = 0.0

    for src, src_lengths, trg, _ in dataloader:
        src = src.to(config.device)
        src_lengths = src_lengths.to(config.device)
        trg = trg.to(config.device)

        outputs = model(
            src=src,
            src_lengths=src_lengths,
            trg=trg,
            teacher_forcing_ratio=0.0,  # 검증 때는 모델 자체 예측 성능만 본다.
        )

        output_dim = outputs.size(-1)

        loss = criterion(
            outputs.reshape(-1, output_dim),
            trg[:, 1:].reshape(-1),
        )

        epoch_loss += loss.item()

    return epoch_loss / max(len(dataloader), 1)


def save_checkpoint(
    model: Seq2Seq,
    vocab: Vocabulary,
    config: Config,
    train_loss: float,
    valid_loss: float,
) -> None:
    """최고 성능 체크포인트를 저장한다."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_stoi": vocab.stoi,
        "vocab_itos": vocab.itos,
        "config": asdict(config),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }
    torch.save(checkpoint, config.checkpoint_path)


def load_checkpoint(model: Seq2Seq, checkpoint_path: str) -> dict:
    """저장된 체크포인트를 불러와 모델 가중치를 복원한다."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def fit(
    model: Seq2Seq,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    vocab: Vocabulary,
    config: Config,
) -> None:
    """
    전체 학습 루프
    - 매 epoch마다 train/valid loss 출력
    - valid loss가 가장 좋을 때 체크포인트 저장
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # PAD는 loss 계산에서 제외해야 한다.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    best_valid_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config)
        valid_loss = evaluate(model, valid_loader, criterion, config)

        train_ppl = math.exp(train_loss) if train_loss < 20 else float("inf")
        valid_ppl = math.exp(valid_loss) if valid_loss < 20 else float("inf")

        print(
            f"[Epoch {epoch:02d}/{config.epochs}] "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | "
            f"train_ppl={train_ppl:.2f} | valid_ppl={valid_ppl:.2f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, vocab, config, train_loss, valid_loss)
            print(f"[INFO] 체크포인트 저장 완료 -> {config.checkpoint_path}")


# ============================================================
# 10. 추론 함수
# ============================================================

@torch.no_grad()
def generate_response(
    model: Seq2Seq,
    sentence: str,
    vocab: Vocabulary,
    config: Config,
    single_stopwords: set[str],
) -> str:
    """
    한 문장을 받아 챗봇 응답을 생성한다.
    - Greedy decoding 사용
    """
    model.eval()

    tokens = tokenize(
        sentence,
        use_stopwords=config.use_stopwords,
        stopwords=single_stopwords,
    )
    tokens = tokens[: config.max_src_len - 1]
    tokens = tokens + [vocab.EOS_TOKEN]

    src_ids = vocab.numericalize(tokens)

    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(config.device)
    src_lengths = torch.tensor([len(src_ids)], dtype=torch.long).to(config.device)

    _, hidden = model.encoder(src_tensor, src_lengths)

    input_token = torch.tensor([vocab.sos_idx], dtype=torch.long).to(config.device)

    generated_ids: List[int] = []

    for _ in range(config.max_decode_len):
        logits, hidden = model.decoder(input_token, hidden)
        next_token_id = logits.argmax(dim=1).item()

        if next_token_id == vocab.eos_idx:
            break

        generated_ids.append(next_token_id)
        input_token = torch.tensor([next_token_id], dtype=torch.long).to(config.device)

    generated_tokens = vocab.denumericalize(generated_ids)
    return detokenize(generated_tokens)


def chat(model: Seq2Seq, vocab: Vocabulary, config: Config, single_stopwords: set[str]) -> None:
    """
    while문 기반 챗봇 인터페이스
    - 'quit', 'exit', '종료' 입력 시 종료
    """
    print("\n[챗봇 시작]")
    print("종료하려면 'quit', 'exit', '종료' 중 하나를 입력하세요.\n")

    while True:
        user_input = input("사용자 > ").strip()

        if user_input.lower() in {"quit", "exit", "종료"}:
            print("챗봇 > 대화를 종료합니다.")
            break

        if not user_input:
            print("챗봇 > 문장을 입력해 주세요.")
            continue

        response = generate_response(model, user_input, vocab, config, single_stopwords)
        print(f"챗봇 > {response}")


# ============================================================
# 11. 전체 파이프라인 조립
# ============================================================

def build_vocab_from_train_df(
    train_df: pd.DataFrame,
    config: Config,
    single_stopwords: set[str],
) -> Vocabulary:
    """
    Vocabulary는 반드시 학습 데이터 기준으로만 만들어야 한다.
    검증/테스트 데이터를 보고 vocab을 만들면 데이터 누수(leakage)가 된다.
    """
    tokenized_texts: List[List[str]] = []

    for text in train_df["Q"].tolist():
        tokenized_texts.append(
            tokenize(text, config.use_stopwords, single_stopwords)
        )

    for text in train_df["A"].tolist():
        tokenized_texts.append(
            tokenize(text, config.use_stopwords, single_stopwords)
        )

    vocab = Vocabulary(min_freq=config.min_freq, max_size=config.max_vocab_size)
    vocab.build(tokenized_texts)
    return vocab


def build_model(vocab_size: int, pad_idx: int, config: Config) -> Seq2Seq:
    """
    Encoder와 Decoder를 만들고 Seq2Seq로 묶는다.
    """
    encoder = Encoder(
        input_dim=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=pad_idx,
        bidirectional=config.bidirectional,
    )

    decoder = Decoder(
        output_dim=vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        pad_idx=pad_idx,
    )

    model = Seq2Seq(encoder, decoder, device=torch.device(config.device)).to(config.device)
    return model


def print_data_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    vocab: Vocabulary,
    config: Config,
    single_stopwords: set[str],
    multi_stopwords: set[str],
) -> None:
    """
    데이터/불용어/Vocabulary 상태를 한 번에 확인하기 위한 요약 출력
    """
    print("\n[데이터 요약]")
    print(f"- train 샘플 수: {len(train_df):,}")
    print(f"- valid 샘플 수: {len(valid_df):,}")
    print(f"- vocab 크기: {len(vocab):,}")
    print(f"- 단일 토큰 불용어 수: {len(single_stopwords):,}")
    print(f"- 다중 어절 불용어 수: {len(multi_stopwords):,}")
    print(f"- use_stopwords: {config.use_stopwords}")
    print(f"- device: {config.device}")
    print(f"- torch_num_threads: {torch.get_num_threads()}")

    print("\n[샘플 미리보기]")
    for i in range(min(3, len(train_df))):
        print(f"Q{i+1}: {train_df.iloc[i]['Q']}")
        print(f"A{i+1}: {train_df.iloc[i]['A']}")
        print("-" * 50)


def main() -> None:
    """
    실행 순서
    1) 시드 고정
    2) 데이터셋 다운로드
    3) 불용어 로딩
    4) train/valid 분할
    5) vocab 생성
    6) dataloader 생성
    7) 모델 생성
    8) 학습
    9) best checkpoint 로드
    10) while문 챗봇 실행
    """
    config = CONFIG
    set_seed(config.seed)
    configure_torch_threads(config.torch_num_threads)

    download_dataset_if_needed(config.dataset_url, config.dataset_path)

    single_stopwords, multi_stopwords = load_stopwords(config.stopwords_path)

    train_df, valid_df = load_dataframe(config, single_stopwords)
    vocab = build_vocab_from_train_df(train_df, config, single_stopwords)
    train_loader, valid_loader = build_dataloaders(
        train_df=train_df,
        valid_df=valid_df,
        vocab=vocab,
        config=config,
        single_stopwords=single_stopwords,
    )

    model = build_model(vocab_size=len(vocab), pad_idx=vocab.pad_idx, config=config)

    print_data_summary(train_df, valid_df, vocab, config, single_stopwords, multi_stopwords)
    print(f"\n[모델 파라미터 수] {count_parameters(model):,}")

    # 이미 저장된 체크포인트가 있으면 그대로 불러오고,
    # 없으면 새로 학습한다.
    if Path(config.checkpoint_path).exists():
        print(f"\n[INFO] 기존 체크포인트를 불러옵니다: {config.checkpoint_path}")
        load_checkpoint(model, config.checkpoint_path)
    else:
        print("\n[INFO] 학습을 시작합니다.")
        fit(model, train_loader, valid_loader, vocab, config)
        print("\n[INFO] 가장 성능이 좋았던 체크포인트를 다시 로드합니다.")
        load_checkpoint(model, config.checkpoint_path)

    chat(model, vocab, config, single_stopwords)


if __name__ == "__main__":
    main()
