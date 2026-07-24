"""图像生成适配器：豆包 Seedream 5.0 Pro (ARK API) - 已修复"""

from __future__ import annotations
import asyncio, base64, hashlib, logging, math, os, random, time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_IMAGE_MODEL = "doubao-seedream-5-0-pro-260628"  # fallback（实际从 config 读取）

_placeholder_count = 0
_total_generations = 0

def get_placeholder_ratio():
    if _total_generations == 0: return 0.0
    return _placeholder_count / _total_generations

def reset_placeholder_stats():
    global _placeholder_count, _total_generations
    _placeholder_count = 0
    _total_generations = 0
IMAGE_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "output" / "images"

@dataclass
class ImageResult:
    url: str
    local_path: str = ""
    minio_path: Optional[str] = None
    prompt: str = ""
    seed: int = 0
    width: int = 1024
    height: int = 1024
    model: str = ""
    cost_usd: float = 0.0
    request_id: str = ""

class ImageAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt="", negative_prompt="", width=1024, height=1024,
                       seed=0, num_images=1, style="", ref_image_url="") -> list[ImageResult]: ...

class FluxAdapter(ImageAdapter):
    @property
    def MODEL(self) -> str:
        """从 config 读取图像模型标识（避免硬编码）。"""
        from app.core.config import settings
        return settings.ARK_IMAGE_MODEL

    def __init__(self, model=""):
        # 优先用传入的 model，其次从 config 读取，最后 fallback 到默认值
        from app.core.config import settings
        self._model = model or settings.ARK_IMAGE_MODEL
        self._http: Optional[httpx.AsyncClient] = None

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=360)
        return self._http

    def _api_key(self) -> str:
        from app.core.config import settings
        return settings.ARK_API_KEY

    async def generate(self, prompt="", negative_prompt="", width=1024, height=1024,
                        seed=0, num_images=1, style="", ref_image_url="") -> list[ImageResult]:
        global _total_generations
        _total_generations += num_images
        actual_seed = seed if seed > 0 else random.randint(0, 2**31-1)
        api_key = self._api_key()
        if not api_key:
            # S7: API key 未配置是配置问题，仍返回 placeholder（但不污染下游，因为 composer 的 Pre-Video Gate 会拦截）
            return self._placeholder(prompt, actual_seed, width, height, num_images)
        w = max(width, 960); h = max(height, 960)
        enhanced = prompt
        if style and "anime" in style.lower():
            enhanced = prompt + ", anime manga style, vibrant colors"
        enhanced = enhanced + ", high quality, detailed"
        body = {"model": self._model, "prompt": enhanced, "size": str(w)+"x"+str(h),
               "n": num_images, "response_format": "url"}
        if ref_image_url:
            body["image"] = ref_image_url
            logger.debug("ARK seedream: using ref image %s", ref_image_url[:60])

        headers = {"Authorization": "Bearer "+api_key, "Content-Type": "application/json"}
        # L0 defence: exponential backoff retry (1s, 3s, 7s)
        http = self._client()
        max_retries = 3
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                resp = await http.post(ARK_BASE+"/images/generations", json=body, headers=headers)
                dt = time.time()-t0
                rid = resp.headers.get("x-request-id","")
                if resp.status_code != 200:
                    last_error = Exception("HTTP {}".format(resp.status_code))
                    if resp.status_code >= 500 and attempt < max_retries:
                        delay = 1.0 * math.pow(3, attempt)
                        logger.warning("L0 retry %d/%d after %.0fs (HTTP %d)", attempt+1, max_retries, delay, resp.status_code)
                        await asyncio.sleep(delay)
                        continue
                    # S7: HTTP 4xx 或 5xx 重试耗尽，抛异常让上层 L1/L2 降级处理
                    logger.error("ARK image HTTP %s: %s", resp.status_code, resp.text[:300])
                    raise RuntimeError("ARK image generation failed: HTTP {}".format(resp.status_code))
                data = resp.json(); results = []
                from app.core.config import settings as _settings
                for i, img in enumerate(data.get("data", [])):
                    img_url = img.get("url","")
                    b64 = img.get("b64_json","")
                    local = await self._save(img_url, b64, actual_seed+i)
                    img_result = ImageResult(url=img_url, local_path=local, prompt=prompt,
                        seed=actual_seed, width=w, height=h, model=self._model, cost_usd=_settings.IMAGE_COST_PER_UNIT, request_id=rid)
                    results.append(img_result)
                    # fire-and-forget MinIO 上传（失败不阻塞管线，minio_path 异步填入）
                    from app.services.minio_client import fire_and_forget_upload
                    fire_and_forget_upload(img_result, local, "image")
                logger.info("ARK seedream: %d images %.1fs", len(results), dt)
                if not results:
                    raise RuntimeError("ARK image generation returned empty results")
                return results
            except RuntimeError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = 1.0 * math.pow(3, attempt)
                    logger.warning("L0 retry %d/%d after %.0fs: %s", attempt+1, max_retries, delay, e)
                    await asyncio.sleep(delay)
                    continue
                logger.error("ARK seedream exhausted retries: %s", e)

        # S7: All retries exhausted — raise instead of returning placeholder
        raise RuntimeError(
            "ARK image generation failed after {} retries: {}".format(max_retries, last_error)
        )

    async def _save(self, url: str, b64: str, seed: int) -> str:
        os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
        fname = "img_"+hashlib.md5((url or b64).encode()).hexdigest()[:12]+"_"+str(seed)+".png"
        local = str(IMAGE_OUTPUT_DIR / fname)
        try:
            if b64:
                if b64.startswith("data:"): b64 = b64.split(",",1)[1]
                with open(local,"wb") as f: f.write(base64.b64decode(b64))
                return local
            if url and url.startswith("http"):
                r = await self._client().get(url)
                if r.status_code == 200:
                    with open(local,"wb") as f: f.write(r.content)
                    logger.debug("Image saved: %s (%d bytes)", local, len(r.content))
                    return local
        except Exception as e:
            logger.warning("Image save failed: %s", e)
        return url or local

    def _placeholder(self, prompt, seed, w, h, n):
        global _placeholder_count
        _placeholder_count += n
        return [ImageResult(url="placeholder://"+hashlib.md5(prompt.encode()).hexdigest()[:16],
                prompt=prompt,seed=seed,width=w,height=h,model=self._model,cost_usd=0.0) for _ in range(n)]

    async def close(self):
        if self._http: await self._http.aclose(); self._http = None

SeedreamAdapter = FluxAdapter
def get_image_adapter(provider="flux") -> ImageAdapter:
    return FluxAdapter()
