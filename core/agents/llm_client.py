"""
LLM 客户端 — OpenAI 兼容协议，支持 GPT / Claude / DeepSeek / 国产模型
对标 Brale: 驱动 Indicator/Pattern/Trend 三个 Agent 的 LLM 推理层
"""
import json
import logging
from typing import Optional

logger = logging.getLogger("picas.llm")

# 延迟导入，避免不使用时加载依赖
_httpx = None


def _get_httpx():
    global _httpx
    if _httpx is None:
        try:
            import httpx
            _httpx = httpx
        except ImportError:
            raise ImportError("请安装 httpx: pip install httpx")
    return _httpx


class LLMClient:
    """OpenAI 兼容协议的 LLM 客户端"""

    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 800)
        self.timeout = config.get("timeout", 30)

        # 清理末尾斜杠
        self.base_url = self.base_url.rstrip("/")

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """
        发送聊天请求

        Args:
            system_prompt: 系统提示词 (Agent 角色定义)
            user_prompt: 用户提示词 (指标数据 + 分析要求)

        Returns:
            LLM 回复文本，失败返回 None
        """
        httpx = _get_httpx()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout + 10,
            )
            if response.status_code != 200:
                logger.error(f"LLM 请求失败: HTTP {response.status_code} {response.text[:200]}")
                return None

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip() if content else None

        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            return None

    def chat_structured(self, system_prompt: str, user_prompt: str,
                        output_format: dict = None) -> dict:
        """
        发送聊天请求并尝试解析结构化 JSON 输出

        Returns:
            {"action": "BUY"|"SELL"|"HOLD", "confidence": float, "reasoning": str, ...}
        """
        # 追加格式要求
        format_instruction = """
请用以下 JSON 格式回复 (不要包含其他内容):
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 ~ 1.0,
  "reasoning": "你的推理过程",
  "signals": ["关键信号1", "关键信号2"],
  "key_level": "关键价位 (如有)"
}
"""
        full_prompt = user_prompt + format_instruction
        raw = self.chat(system_prompt, full_prompt)

        if raw is None:
            return {"action": "HOLD", "confidence": 0.0,
                    "reasoning": "LLM调用失败", "signals": [], "key_level": ""}

        # 尝试解析 JSON
        try:
            # 清理可能的 markdown 代码块包裹
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
                text = text.strip()
            result = json.loads(text)
            return {
                "action": result.get("action", "HOLD").upper(),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", raw),
                "signals": result.get("signals", []),
                "key_level": result.get("key_level", ""),
            }
        except (json.JSONDecodeError, ValueError):
            # JSON 解析失败，基于文本启发式提取
            return self._fallback_parse(raw)

    def _fallback_parse(self, text: str) -> dict:
        """JSON 解析失败时的启发式解析"""
        text_upper = text.upper()
        if "BUY" in text_upper and "SELL" not in text_upper:
            action, conf = "BUY", 0.6
        elif "SELL" in text_upper and "BUY" not in text_upper:
            action, conf = "SELL", 0.6
        else:
            action, conf = "HOLD", 0.4

        return {
            "action": action,
            "confidence": conf,
            "reasoning": text[:300],
            "signals": [],
            "key_level": "",
        }


# ---------------------------------------------------------------------------
# 全局 LLM 客户端 (单例)
# ---------------------------------------------------------------------------
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端"""
    global _llm_client
    if _llm_client is None:
        from config.loader import get_llm_config
        cfg = get_llm_config()
        _llm_client = LLMClient(cfg)
        logger.info(f"LLM客户端初始化: model={_llm_client.model}")
    return _llm_client
