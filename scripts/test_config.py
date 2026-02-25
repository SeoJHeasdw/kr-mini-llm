#!/usr/bin/env python3
"""
설정 파일 테스트 스크립트

M4 Max 최적화 설정이 제대로 로드되는지 확인합니다.
"""

from pathlib import Path
import sys

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.config import TransformerConfig


def test_config_loading():
    """설정 파일 로드 테스트"""
    print("="*60)
    print("🧪 설정 파일 로드 테스트")
    print("="*60)
    
    configs = [
        ("configs/model_small.yaml", "Small (레거시)"),
        ("configs/model_medium.yaml", "Medium (권장)"),
        ("configs/model_large.yaml", "Large (도전)"),
    ]
    
    for config_path, name in configs:
        print(f"\n📄 {name}: {config_path}")
        try:
            config = TransformerConfig.from_yaml(config_path)
            print(f"   ✅ 로드 성공")
            print(f"   - Hidden size: {config.hidden_size}")
            print(f"   - Layers: {config.num_layers}")
            print(f"   - Heads: {config.num_heads}")
            print(f"   - Parameters: ~{config.num_parameters/1e6:.1f}M")
        except Exception as e:
            print(f"   ❌ 로드 실패: {e}")
            return False
    
    return True


def test_presets():
    """프리셋 테스트"""
    print("\n" + "="*60)
    print("🧪 프리셋 테스트")
    print("="*60)
    
    presets = [
        ("small", TransformerConfig.small()),
        ("medium", TransformerConfig.medium()),
        ("large", TransformerConfig.large()),
    ]
    
    for name, config in presets:
        print(f"\n🎯 {name.upper()} 프리셋:")
        print(config)
    
    return True


def test_config_methods():
    """설정 메서드 테스트"""
    print("\n" + "="*60)
    print("🧪 설정 메서드 테스트")
    print("="*60)
    
    # Medium 설정 생성
    config = TransformerConfig.medium()
    
    # to_dict 테스트
    print("\n📋 to_dict():")
    config_dict = config.to_dict()
    print(f"   Keys: {list(config_dict.keys())}")
    print(f"   ✅ 성공")
    
    # save_yaml 테스트
    print("\n💾 save_yaml():")
    test_path = Path("test_config_output.yaml")
    try:
        config.save_yaml(test_path)
        print(f"   ✅ 저장 성공: {test_path}")
        
        # 다시 로드해서 확인
        loaded_config = TransformerConfig.from_yaml(test_path)
        if loaded_config.hidden_size == config.hidden_size:
            print(f"   ✅ 로드 검증 성공")
        else:
            print(f"   ❌ 로드 검증 실패")
            return False
        
        # 테스트 파일 삭제
        test_path.unlink()
        print(f"   🗑️  테스트 파일 삭제")
        
    except Exception as e:
        print(f"   ❌ 실패: {e}")
        return False
    
    return True


def main():
    """메인 테스트 실행"""
    print("\n" + "🚀 "*30)
    print("kr-mini-llm 설정 테스트")
    print("🚀 "*30 + "\n")
    
    tests = [
        ("설정 파일 로드", test_config_loading),
        ("프리셋", test_presets),
        ("설정 메서드", test_config_methods),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 테스트 중 예외 발생: {e}")
            results.append((name, False))
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️  일부 테스트 실패")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
