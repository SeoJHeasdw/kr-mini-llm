from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class TransformerConfig:
    """
    Transformer 언어모델 설정값.
    
    M4 Max 최적화 설정 (Medium: 350M, Large: 800M 파라미터)
    """

    vocab_size: int = 32000
    hidden_size: int = 1024  # Medium: 1024, Large: 1536
    num_layers: int = 24     # Medium/Large: 24
    num_heads: int = 16      # Medium: 16, Large: 24
    num_kv_heads: int = 4    # GQA용 (Medium: 4, Large: 6)
    intermediate_size: int = 4096  # SwiGLU용 (4 * hidden_size)
    max_seq_length: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TransformerConfig":
        """
        YAML 파일에서 설정 로드
        
        Args:
            path: YAML 설정 파일 경로
            
        Returns:
            TransformerConfig 인스턴스
            
        Example:
            >>> config = TransformerConfig.from_yaml("configs/model_medium.yaml")
        """
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # YAML에서 주석이나 추가 필드 제거
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_config)
    
    @classmethod
    def small(cls) -> "TransformerConfig":
        """
        Small 모델 프리셋 (레거시 - 테스트용)
        파라미터: ~50M
        """
        return cls(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            intermediate_size=2048,
            max_seq_length=1024,
            dropout=0.0
        )
    
    @classmethod
    def medium(cls) -> "TransformerConfig":
        """
        Medium 모델 프리셋 (M4 Max 권장)
        파라미터: ~350M
        메모리: ~10GB (학습 시)
        """
        return cls(
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            num_kv_heads=4,
            intermediate_size=4096,
            max_seq_length=2048,
            dropout=0.1
        )
    
    @classmethod
    def large(cls) -> "TransformerConfig":
        """
        Large 모델 프리셋 (M4 Max 도전)
        파라미터: ~800M
        메모리: ~20GB (학습 시)
        """
        return cls(
            hidden_size=1536,
            num_layers=24,
            num_heads=24,
            num_kv_heads=6,
            intermediate_size=6144,
            max_seq_length=2048,
            dropout=0.1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'intermediate_size': self.intermediate_size,
            'max_seq_length': self.max_seq_length,
            'rope_theta': self.rope_theta,
            'dropout': self.dropout,
        }
    
    def save_yaml(self, path: str | Path) -> None:
        """설정을 YAML 파일로 저장"""
        import yaml
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @property
    def num_parameters(self) -> int:
        """
        대략적인 파라미터 수 계산
        
        Returns:
            예상 파라미터 수
        """
        # Embedding
        embed_params = self.vocab_size * self.hidden_size
        
        # Transformer blocks
        # Attention: Q, K, V projections + output projection
        attn_params = (
            self.hidden_size * self.hidden_size * 3 +  # Q, K, V
            self.hidden_size * self.hidden_size         # output
        ) * self.num_layers
        
        # FFN: gate, up, down projections
        ffn_params = (
            self.hidden_size * self.intermediate_size * 3  # gate, up, down
        ) * self.num_layers
        
        # Layer norms (2 per block)
        norm_params = self.hidden_size * 2 * self.num_layers
        
        # Output head
        output_params = self.hidden_size * self.vocab_size
        
        total = embed_params + attn_params + ffn_params + norm_params + output_params
        return total
    
    def __str__(self) -> str:
        """설정 정보를 문자열로 반환"""
        params = self.num_parameters / 1e6  # 백만 단위
        return (
            f"TransformerConfig(\n"
            f"  vocab_size={self.vocab_size:,},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  num_kv_heads={self.num_kv_heads},\n"
            f"  intermediate_size={self.intermediate_size},\n"
            f"  max_seq_length={self.max_seq_length},\n"
            f"  rope_theta={self.rope_theta},\n"
            f"  dropout={self.dropout},\n"
            f"  ~{params:.1f}M parameters\n"
            f")"
        )


