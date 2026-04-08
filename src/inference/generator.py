"""
텍스트 생성 모듈
Korean LLM을 위한 텍스트 생성 및 샘플링 전략 구현
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """텍스트 생성 설정"""
    max_length: int = 512
    max_new_tokens: Optional[int] = None
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    
    def __post_init__(self):
        """설정 검증"""
        if self.temperature <= 0:
            raise ValueError(f"temperature는 0보다 커야 합니다: {self.temperature}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError(f"top_k는 0보다 커야 합니다: {self.top_k}")
        if self.top_p is not None and not (0 < self.top_p <= 1):
            raise ValueError(f"top_p는 0과 1 사이여야 합니다: {self.top_p}")
        if self.repetition_penalty < 0:
            raise ValueError(f"repetition_penalty는 0 이상이어야 합니다: {self.repetition_penalty}")


class Generator:
    """
    텍스트 생성기
    
    다양한 샘플링 전략 지원:
    - Greedy decoding (temperature=0 또는 do_sample=False)
    - Temperature sampling
    - Top-K sampling
    - Top-P (nucleus) sampling
    - Beam search (num_beams > 1)
    - Repetition penalty
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: TransformerLM 모델
            tokenizer: 토크나이저
            device: 실행 디바이스
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Generator 초기화 완료 (device: {self.device})")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            config: 생성 설정
            **kwargs: GenerationConfig 오버라이드
            
        Returns:
            생성된 텍스트
        """
        # 설정 준비
        if config is None:
            config = GenerationConfig(**kwargs)
        else:
            # kwargs로 오버라이드
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 토크나이징
        input_ids = self._encode(prompt)
        
        # 생성
        if config.num_beams > 1:
            output_ids = self._beam_search(input_ids, config)
        else:
            output_ids = self._sample(input_ids, config)
        
        # 디코딩
        generated_text = self._decode(output_ids)
        
        return generated_text
    
    def _encode(self, text: str) -> torch.Tensor:
        """텍스트를 토큰 ID로 변환"""
        # 토크나이저가 encode 메서드를 가지고 있다고 가정
        if hasattr(self.tokenizer, 'encode'):
            token_ids = self.tokenizer.encode(text)
        else:
            # 간단한 fallback
            token_ids = [ord(c) for c in text]
        
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)
    
    def _decode(self, token_ids: torch.Tensor) -> str:
        """토큰 ID를 텍스트로 변환"""
        # 배치 차원 제거
        if token_ids.dim() > 1:
            token_ids = token_ids[0]
        
        # 토크나이저가 decode 메서드를 가지고 있다고 가정
        if hasattr(self.tokenizer, 'decode'):
            text = self.tokenizer.decode(token_ids.tolist())
        else:
            # 간단한 fallback
            text = ''.join(chr(id) for id in token_ids.tolist() if id < 128)
        
        return text
    
    def _sample(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        자기회귀 샘플링
        
        Args:
            input_ids: (1, seq_len) 입력 토큰
            config: 생성 설정
            
        Returns:
            (1, total_len) 생성된 토큰
        """
        batch_size = input_ids.size(0)
        cur_len = input_ids.size(1)
        
        # 최대 길이 결정
        if config.max_new_tokens is not None:
            max_length = cur_len + config.max_new_tokens
        else:
            max_length = config.max_length
        
        # 생성된 토큰 추적 (repetition penalty용)
        generated_tokens = input_ids.clone()
        
        # KV cache 초기화
        kv_caches = None
        use_cache = True
        
        while cur_len < max_length:
            # Forward pass
            if use_cache and kv_caches is not None:
                # 캐시 사용: 마지막 토큰만 처리
                model_input = input_ids[:, -1:]
                start_pos = cur_len - 1
            else:
                # 캐시 없음: 전체 시퀀스 처리
                model_input = input_ids
                start_pos = 0
            
            # 모델 실행
            logits, kv_caches = self.model(
                model_input,
                kv_caches=kv_caches,
                use_cache=use_cache,
                start_pos=start_pos
            )
            
            # 마지막 토큰의 logits
            next_token_logits = logits[:, -1, :]
            
            # Repetition penalty 적용
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated_tokens,
                    config.repetition_penalty
                )
            
            # 다음 토큰 샘플링
            if config.do_sample and config.temperature > 0:
                next_token = self._sample_next_token(
                    next_token_logits,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p
                )
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 토큰 추가
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            cur_len += 1
            
            # EOS 토큰 체크
            if config.eos_token_id is not None:
                if (next_token == config.eos_token_id).all():
                    if config.early_stopping:
                        break
        
        return input_ids
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        다음 토큰 샘플링
        
        Args:
            logits: (batch, vocab_size) 로짓
            temperature: 샘플링 온도
            top_k: Top-K 필터링
            top_p: Top-P (nucleus) 필터링
            
        Returns:
            (batch, 1) 샘플링된 토큰
        """
        # Temperature 적용
        logits = logits / temperature
        
        # Top-K 필터링
        if top_k is not None and top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        
        # Top-P 필터링
        if top_p is not None and top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)
        
        # 확률 분포로 변환
        probs = F.softmax(logits, dim=-1)
        
        # 샘플링
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """
        Top-K 필터링
        
        상위 K개 토큰만 유지하고 나머지는 -inf로 설정
        """
        top_k = min(top_k, logits.size(-1))
        
        # 상위 K개 값 찾기
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1:].expand_as(logits)
        
        # K 밖의 값들을 -inf로 설정
        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float('-inf')),
            logits
        )
        
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """
        Top-P (nucleus) 필터링
        
        누적 확률이 top_p를 초과하는 토큰들만 유지
        """
        # 내림차순 정렬
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # 누적 확률 계산
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # top_p를 초과하는 토큰 제거
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 첫 번째 토큰은 항상 유지 (최소 1개는 선택되도록)
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # 원래 순서로 복원
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """
        Repetition penalty 적용
        
        이미 생성된 토큰의 확률을 감소시킴
        """
        if penalty == 1.0:
            return logits
        
        # 생성된 토큰들의 로짓 조정
        for token_id in generated_tokens[0].unique():
            # penalty > 1: 확률 감소
            # penalty < 1: 확률 증가
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        
        return logits
    
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        Beam search 생성
        
        Args:
            input_ids: (1, seq_len) 입력 토큰
            config: 생성 설정
            
        Returns:
            (1, total_len) 생성된 토큰
        """
        batch_size = input_ids.size(0)
        num_beams = config.num_beams
        vocab_size = int(self.model.cfg.vocab_size)
        
        # 최대 길이 결정
        if config.max_new_tokens is not None:
            max_length = input_ids.size(1) + config.max_new_tokens
        else:
            max_length = config.max_length
        
        # Beam 초기화: (batch * num_beams, seq_len)
        beam_input_ids = input_ids.repeat(num_beams, 1)
        beam_scores = torch.zeros(batch_size * num_beams, device=self.device)
        
        # 완료된 beam 추적
        done_beams = []
        
        cur_len = input_ids.size(1)
        
        while cur_len < max_length:
            # Forward pass
            logits, _ = self.model(beam_input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Length penalty 적용
            if config.length_penalty != 1.0:
                next_token_scores = next_token_scores / (cur_len ** config.length_penalty)
            
            # Beam scores 업데이트
            next_scores = beam_scores.unsqueeze(1) + next_token_scores
            
            # Reshape: (batch, num_beams * vocab_size)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # 상위 num_beams개 선택
            next_scores, next_tokens = torch.topk(
                next_scores, num_beams, dim=1, largest=True, sorted=True
            )
            
            # Beam index와 token index 계산
            next_beam_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Beam 업데이트
            beam_input_ids = torch.cat([
                beam_input_ids[next_beam_indices.view(-1)],
                next_tokens.view(-1, 1)
            ], dim=1)
            beam_scores = next_scores.view(-1)
            
            cur_len += 1
            
            # EOS 토큰 체크
            if config.eos_token_id is not None:
                eos_mask = next_tokens == config.eos_token_id
                if eos_mask.any() and config.early_stopping:
                    # 완료된 beam 저장
                    for i in range(batch_size):
                        for j in range(num_beams):
                            if eos_mask[i, j]:
                                done_beams.append({
                                    'tokens': beam_input_ids[i * num_beams + j],
                                    'score': beam_scores[i * num_beams + j]
                                })
                    
                    if len(done_beams) >= num_beams:
                        break
        
        # 최고 점수 beam 반환
        if done_beams:
            best_beam = max(done_beams, key=lambda x: x['score'])
            return best_beam['tokens'].unsqueeze(0)
        else:
            # 첫 번째 beam 반환
            return beam_input_ids[:1]
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[str]:
        """
        배치 텍스트 생성
        
        Args:
            prompts: 입력 프롬프트 리스트
            config: 생성 설정
            **kwargs: GenerationConfig 오버라이드
            
        Returns:
            생성된 텍스트 리스트
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, config, **kwargs)
            results.append(result)
        return results



