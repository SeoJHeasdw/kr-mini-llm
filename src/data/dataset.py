"""
데이터셋 클래스 정의

TextDataset: 텍스트 파일을 직접 로드 (메모리 효율적이지 않음)
TokenizedDataset: 미리 토큰화된 .npy 파일 로드 (권장)
StreamingTextDataset: 대용량 텍스트 파일을 스트리밍으로 로드
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List
import numpy as np

from .tokenizer import KoreanTokenizer


class TokenizedDataset(Dataset):
    """
    미리 토큰화된 numpy 파일을 로드하는 데이터셋 (권장)
    
    scripts/tokenize_data.py로 미리 토큰화한 .npy 파일 사용
    학습 시 토큰화 오버헤드 없이 빠르게 로드 가능
    """
    
    def __init__(
        self,
        tokenized_file: str,
        max_length: int = 2048,
    ):
        """
        Args:
            tokenized_file: 토큰화된 numpy 파일 (.npy)
            max_length: 최대 시퀀스 길이
        """
        self.max_length = max_length
        
        print(f"📂 토큰화된 데이터 로딩: {tokenized_file}")
        
        # numpy 파일 로드
        self.sequences = np.load(tokenized_file, allow_pickle=True)
        
        print(f"   ✅ 로드 완료: {len(self.sequences):,} 시퀀스")
        
        # 통계 출력
        lengths = [len(seq) for seq in self.sequences]
        print(f"   - 평균 길이: {np.mean(lengths):.1f}")
        print(f"   - 최소 길이: {np.min(lengths)}")
        print(f"   - 최대 길이: {np.max(lengths)}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        시퀀스 반환
        
        Returns:
            tokens: [seq_len] 토큰 ID 텐서
        """
        tokens = self.sequences[idx]
        
        # max_length로 자르기
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return torch.tensor(tokens, dtype=torch.long)


class TextDataset(Dataset):
    """
    텍스트 파일을 직접 로드하는 데이터셋
    
    주의: 학습 시마다 토큰화하므로 느림
    작은 데이터셋이나 테스트용으로만 사용 권장
    """
    
    def __init__(
        self,
        text_file: str,
        tokenizer: KoreanTokenizer,
        max_length: int = 2048,
        stride: int = 1024,
    ):
        """
        Args:
            text_file: 텍스트 파일 경로
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
            stride: 슬라이딩 윈도우 스트라이드
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        print(f"📖 텍스트 파일 로딩: {text_file}")
        
        # 전체 텍스트 로드
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"   - 총 라인 수: {len(lines):,}")
        
        # 토큰화 및 청크 생성
        self.sequences = []
        total_tokens = 0
        
        for line in lines:
            tokens = tokenizer.encode(line)
            total_tokens += len(tokens)
            
            # 슬라이딩 윈도우로 청크 생성
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + max_length]
                if len(chunk) >= 10:  # 최소 길이
                    self.sequences.append(chunk)
        
        print(f"   ✅ 로드 완료: {len(self.sequences):,} 시퀀스")
        print(f"   - 총 토큰 수: {total_tokens:,}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        시퀀스 반환
        
        Returns:
            tokens: [seq_len] 토큰 ID 텐서
        """
        tokens = self.sequences[idx]
        return torch.tensor(tokens, dtype=torch.long)


class StreamingTextDataset(Dataset):
    """
    대용량 텍스트 파일을 스트리밍으로 로드하는 데이터셋
    
    메모리에 전체 데이터를 올리지 않고 필요할 때마다 읽음
    매우 큰 데이터셋에 유용하지만 느림
    """
    
    def __init__(
        self,
        text_file: str,
        tokenizer: KoreanTokenizer,
        max_length: int = 2048,
    ):
        """
        Args:
            text_file: 텍스트 파일 경로
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.text_file = text_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 라인 오프셋 계산
        print(f"📂 파일 인덱싱: {text_file}")
        self.line_offsets = [0]
        
        with open(text_file, 'rb') as f:
            while f.readline():
                self.line_offsets.append(f.tell())
        
        # 마지막 오프셋 제거 (EOF)
        self.line_offsets.pop()
        
        print(f"   ✅ 인덱싱 완료: {len(self.line_offsets):,} 라인")
    
    def __len__(self) -> int:
        return len(self.line_offsets)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        시퀀스 반환 (파일에서 직접 읽음)
        
        Returns:
            tokens: [seq_len] 토큰 ID 텐서
        """
        # 파일에서 해당 라인 읽기
        with open(self.text_file, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().strip()
        
        # 토큰화
        tokens = self.tokenizer.encode(line)
        
        # max_length로 자르기
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> dict:
    """
    배치 생성 함수
    
    가변 길이 시퀀스를 패딩하여 동일한 길이로 만듦
    
    Args:
        batch: 토큰 시퀀스 리스트
    
    Returns:
        dict with keys:
            - input_ids: [batch_size, max_seq_len] 입력 토큰
            - attention_mask: [batch_size, max_seq_len] 어텐션 마스크
            - labels: [batch_size, max_seq_len] 레이블 (input_ids와 동일)
    """
    # 최대 길이 찾기
    max_len = max(len(seq) for seq in batch)
    
    # 패딩
    input_ids = []
    attention_mask = []
    
    for seq in batch:
        seq_len = len(seq)
        
        # 패딩 추가
        padded = torch.cat([
            seq,
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        
        # 어텐션 마스크 (실제 토큰은 1, 패딩은 0)
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        
        input_ids.append(padded)
        attention_mask.append(mask)
    
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    
    # 레이블은 input_ids와 동일 (언어 모델링)
    labels = input_ids.clone()
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def create_dataloaders(
    train_data: str,
    valid_data: str,
    tokenizer: Optional[KoreanTokenizer] = None,
    batch_size: int = 8,
    max_length: int = 2048,
    num_workers: int = 4,
    use_tokenized: bool = True,
    streaming: bool = False,
) -> tuple:
    """
    학습 및 검증 데이터로더 생성
    
    Args:
        train_data: 학습 데이터 파일 (.npy 또는 .txt)
        valid_data: 검증 데이터 파일 (.npy 또는 .txt)
        tokenizer: 토크나이저 (텍스트 파일 사용 시 필요)
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        num_workers: 데이터 로딩 워커 수
        use_tokenized: 토큰화된 .npy 파일 사용 여부
        streaming: 스트리밍 모드 사용 여부 (텍스트 파일만)
    
    Returns:
        train_loader: 학습 데이터로더
        valid_loader: 검증 데이터로더
    """
    # 데이터셋 생성
    if use_tokenized:
        # 토큰화된 파일 사용 (권장)
        print("🚀 토큰화된 데이터 사용 (빠름)")
        train_dataset = TokenizedDataset(train_data, max_length)
        valid_dataset = TokenizedDataset(valid_data, max_length)
    elif streaming:
        # 스트리밍 모드
        print("🌊 스트리밍 모드 사용 (메모리 효율적)")
        if tokenizer is None:
            raise ValueError("스트리밍 모드는 tokenizer가 필요합니다")
        train_dataset = StreamingTextDataset(train_data, tokenizer, max_length)
        valid_dataset = StreamingTextDataset(valid_data, tokenizer, max_length)
    else:
        # 텍스트 파일 직접 로드
        print("📖 텍스트 파일 직접 로드 (느림)")
        if tokenizer is None:
            raise ValueError("텍스트 파일 로드는 tokenizer가 필요합니다")
        train_dataset = TextDataset(train_data, tokenizer, max_length)
        valid_dataset = TextDataset(valid_data, tokenizer, max_length)
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, valid_loader


