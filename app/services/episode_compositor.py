"""Episode Compositor — FFmpeg final composition.

Takes per-scene video segments + per-scene audio (Seed Audio 1.0 输出，
已整合对白 + BGM + 音效) and produces a single final MP4 with:
  1. Video segments concatenated in scene order
  2. Audio mixed in at the correct timestamps (synced to each scene)
  3. Optional legacy BGM bed at low volume (bgm_path 非空时，向后兼容)

Uses imageio-ffmpeg to locate the ffmpeg binary so no system install is needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"


def _get_ffmpeg() -> str:
    """Return path to ffmpeg binary via imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        logger.warning("imageio-ffmpeg not available: %s, falling back to system ffmpeg", e)
        return "ffmpeg"


def _get_ffprobe() -> Optional[str]:
    """Return path to ffprobe binary, deriving from ffmpeg path.

    借鉴 QinKunming/ai_manju video_composer.py 的 _get_ffprobe_path：
    从 ffmpeg 路径推导 ffprobe 路径，若不存在返回 None。

    注意：imageio_ffmpeg 提供的 ffmpeg 路径格式为 ffmpeg-win-x86_64-v7.1.exe，
    不包含 "ffmpeg.exe" 后缀，需要用文件名替换而非全路径替换。
    且 imageio_ffmpeg 不提供 ffprobe，所以通常返回 None，fallback 到 stderr 解析。
    """
    ffmpeg = _get_ffmpeg()
    ffmpeg_dir = os.path.dirname(ffmpeg)
    ffmpeg_name = os.path.basename(ffmpeg)
    # 把文件名中的 "ffmpeg" 替换为 "ffprobe"（不是替换全路径）
    ffprobe_name = ffmpeg_name.replace("ffmpeg", "ffprobe")
    if ffprobe_name != ffmpeg_name:  # 确保替换生效
        ffprobe_path = os.path.join(ffmpeg_dir, ffprobe_name)
        if os.path.isfile(ffprobe_path):
            return ffprobe_path
    # imageio_ffmpeg 不提供 ffprobe，返回 None（fallback 到 stderr 解析）
    return None


def _run_ffmpeg(args: list[str], timeout: int = 300) -> tuple[bool, str]:
    """Run ffmpeg with given args, return (success, stderr_output)."""
    ffmpeg = _get_ffmpeg()
    cmd = [ffmpeg] + args
    logger.info("ffmpeg: %s", " ".join(cmd[:6]) + " ...")
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", "ignore")[-800:]
            logger.error("ffmpeg failed (code %d): %s", r.returncode, err)
            return False, err
        return True, r.stderr.decode("utf-8", "ignore")[-400:]
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timeout after %ds", timeout)
        return False, "timeout"
    except Exception as e:
        logger.error("ffmpeg exception: %s", e)
        return False, str(e)


def _concat_audio_list(paths: list[str], tmp_dir: Path, scene_idx: int) -> str:
    """Concatenate multiple audio files into one using ffmpeg concat demuxer."""
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]

    list_path = str(tmp_dir / f"audio_list_s{scene_idx}.txt")
    # NOTE: 扩展名 .m4a（AAC 容器）；用 -c:a aac 重新编码以兼容混合输入格式
    # （timeline 模式输出 .m4a，legacy TTS 模式输出 .mp3；统一转 AAC 避免容器冲突）
    out_path = str(tmp_dir / f"audio_concat_s{scene_idx}.m4a")

    with open(list_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(f"file '{p.replace(chr(92), '/')}'\n")

    args = ["-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", out_path]
    ok, _ = _run_ffmpeg(args, timeout=60)

    try:
        os.remove(list_path)
    except Exception:
        pass

    if ok and os.path.isfile(out_path):
        logger.debug("Audio concatenated: %d files -> %s", len(paths), out_path)
        return out_path
    return paths[0]


def _probe_has_audio(path: str) -> bool:
    """Check if a video file has an audio stream.

    借鉴 QinKunming/ai_manju video_composer.py 的 _has_audio_stream：
    优先用 ffprobe（标准、可靠），fallback 到 ffmpeg stderr 解析。
    """
    ffprobe = _get_ffprobe()
    if ffprobe:
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            )
            return bool(r.stdout.strip())
        except Exception as e:
            logger.debug("ffprobe audio probe failed for %s: %s", path, e)
    # Fallback: parse ffmpeg stderr
    ffmpeg = _get_ffmpeg()
    try:
        r = subprocess.run(
            [ffmpeg, "-i", path],
            capture_output=True, timeout=15,
        )
        stderr = r.stderr.decode("utf-8", "ignore")
        return "Audio:" in stderr
    except Exception as e:
        logger.debug("Audio probe failed for %s: %s", path, e)
        return False


def _probe_video_duration(path: str) -> float:
    """Probe actual video file duration using ffmpeg (reliable method)."""
    ffmpeg = _get_ffmpeg()
    try:
        # Use -f null - to force ffmpeg to fully parse the file
        # This is slower than -i alone but much more reliable
        r = subprocess.run(
            [ffmpeg, "-i", path, "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
        stderr = r.stderr.decode("utf-8", "ignore")
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
        if m:
            dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            if dur > 0:
                return dur
        # Fallback: try without -f null (just parse header)
        r2 = subprocess.run(
            [ffmpeg, "-i", path],
            capture_output=True, timeout=10,
        )
        stderr2 = r2.stderr.decode("utf-8", "ignore")
        m2 = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr2)
        if m2:
            return int(m2.group(1)) * 3600 + int(m2.group(2)) * 60 + float(m2.group(3))
    except Exception as e:
        logger.debug("Probe failed for %s: %s", path, e)
    return 0.0


def _probe_audio_duration(path: str) -> float:
    """Probe audio file duration using ffmpeg."""
    ffmpeg = _get_ffmpeg()
    try:
        import re
        r = subprocess.run(
            [ffmpeg, "-i", path, "-f", "null", "-"],
            capture_output=True, timeout=30,
        )
        stderr = r.stderr.decode("utf-8", "ignore")
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
        if m:
            dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            if dur > 0:
                return dur
    except Exception as e:
        logger.debug("Audio probe failed for %s: %s", path, e)
    # Fallback 1: try mutagen (lightweight, no subprocess)
    try:
        from mutagen.mp3 import MP3
        audio = MP3(path)
        if audio.info.length > 0:
            return audio.info.length
    except Exception:
        pass
    # Fallback 2: estimate from file size (mp3 ~16KB/s at 128kbps)
    try:
        size_bytes = os.path.getsize(path)
        estimated_s = size_bytes / (128 * 1000 / 8)  # 128kbps CBR estimate
        if estimated_s > 0.5:
            logger.debug("Audio duration estimated from file size: %.1fs", estimated_s)
            return estimated_s
    except Exception:
        pass
    return 0.0


def _normalize_clip(video_path: str, output_path: str) -> bool:
    """归一化单个clip：确保有音频流 + 统一编码参数（为 concat demuxer 准备）.

    借鉴 QinKunming/ai_manju video_composer.py 的 per-shot 混合策略：
    无音频流的clip用 anullsrc 生成静音，确保所有clip格式一致，
    避免 concat demuxer 因音频流缺失/参数不一致而失败。
    """
    if not os.path.isfile(video_path):
        logger.warning("Normalize: video file missing: %s", video_path)
        return False

    has_audio = _probe_has_audio(video_path)

    if has_audio:
        args = [
            "-y",
            "-i", video_path,
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-ac", "2",
            "-r", "30",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        # 无音频流：用 anullsrc 生成静音，-map 0:v -map 1:a 限定流
        dur = _probe_video_duration(video_path)
        if dur < 0.5:
            dur = 5.0
        logger.info("Normalize: clip has no audio, adding silence (%.2fs): %s", dur, video_path)
        args = [
            "-y",
            "-i", video_path,
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-r", "30",
            "-t", f"{dur:.3f}",
            "-movflags", "+faststart",
            output_path,
        ]

    ok, err = _run_ffmpeg(args, timeout=120)
    if not ok:
        logger.warning("Normalize failed for %s: %s", video_path, err[:200])
    return ok


def _concatenate_clips(clip_paths: list[str], output_path: str, clip_durations=None, crossfade_dur: float = 0.5) -> bool:
    """Concatenate video clips using concat demuxer (per-shot 归一化策略).

    借鉴 QinKunming/ai_manju video_composer.py：
    放弃 xfade 转场（过于复杂、对音频流一致性要求苛刻），
    改用 concat demuxer + per-shot 归一化：
      1. 每个 clip 先归一化（确保有音频流 + 统一编码参数）
      2. 用 concat demuxer 拼接
    这彻底避免 "Could not open encoder before EOF" 等音频流不一致错误。

    Args:
        clip_paths: Ordered list of video file paths.
        output_path: Output file path.
        clip_durations: 保留参数兼容性（不再用于 xfade offset 计算）。
        crossfade_dur: 保留参数兼容性（不再使用 xfade）。
    """
    if not clip_paths:
        return False
    if len(clip_paths) == 1:
        import shutil
        shutil.copy2(clip_paths[0], output_path)
        return True

    tmp_dir = Path(output_path).parent
    n = len(clip_paths)

    # 探测各clip时长（用于日志）
    durations = []
    for cp in clip_paths:
        d = _probe_video_duration(cp)
        durations.append(d if d > 0.5 else 5.0)
    logger.info("Concat %d clips, probed durations: %s", n, [f"{d:.1f}s" for d in durations])
    logger.info("Expected output duration: %.1fs (no xfade overlap)", sum(durations))

    # Step 1: 归一化每个clip（确保有音频流 + 统一编码参数）
    normalized = []
    norm_failed = False
    for i, cp in enumerate(clip_paths):
        norm_path = str(tmp_dir / f"norm_{i:03d}.mp4")
        if _normalize_clip(cp, norm_path) and os.path.isfile(norm_path):
            normalized.append(norm_path)
        else:
            # 归一化失败：用原始文件（concat demuxer 可能仍能处理）
            logger.warning("Normalize clip %d failed, using raw: %s", i, cp)
            normalized.append(cp)
            norm_failed = True

    # Step 2: concat demuxer 拼接
    list_path = str(tmp_dir / "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for cp in normalized:
            f.write(f"file '{cp.replace(chr(92), '/')}'\n")

    # 重新编码（-c:v libx264）而非 -c copy，确保参数完全一致
    # copy 模式对参数一致性要求极高，归一化后理论上可以 copy，
    # 但为最大稳定性，统一重新编码
    # -map 0:v:0 -map 0:a:0? 明确映射视频和音频流，防止音频丢失
    # -ac 2 统一声道为 stereo，防止 mono/stereo 混合导致 concat 丢音频
    args = [
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-ac", "2",
        "-r", "30",
        "-movflags", "+faststart",
        output_path,
    ]
    ok, err = _run_ffmpeg(args, timeout=300)

    # 清理归一化临时文件
    for i, norm_path in enumerate(normalized):
        if norm_path != clip_paths[i]:  # 不要删原始文件
            try:
                os.remove(norm_path)
            except Exception:
                pass
    try:
        os.remove(list_path)
    except Exception:
        pass

    if ok and os.path.isfile(output_path):
        actual_dur = _probe_video_duration(output_path)
        size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("Concat %d clips (demuxer) -> %s (%.1fs, %.1f MB)",
                    n, output_path, actual_dur, size)
        if actual_dur < sum(durations) * 0.7:
            logger.warning("Concat output too short (%.1fs < %.1fs*0.7)",
                           actual_dur, sum(durations))
        return True

    logger.error("Concat demuxer failed: %s", err[:300])

    # 最终 fallback: 尝试 -c copy 模式（不重新编码）
    if norm_failed:
        logger.warning("Trying concat demuxer with -c copy (raw clips)")
        with open(list_path, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                f.write(f"file '{cp.replace(chr(92), '/')}'\n")
        args2 = ["-y", "-f", "concat", "-safe", "0", "-i", list_path,
                 "-c", "copy", "-movflags", "+faststart", output_path]
        ok2, err2 = _run_ffmpeg(args2, timeout=120)
        try:
            os.remove(list_path)
        except Exception:
            pass
        if ok2 and os.path.isfile(output_path):
            size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info("Concat %d clips (copy fallback) -> %s (%.1f MB)", n, output_path, size)
            return True
        logger.error("Concat copy fallback also failed: %s", err2[:200])

    return False


def _mix_bgm(video_path: str, bgm_path: str, output_path: str, bgm_vol: float = 0.15) -> bool:
    """Mix a background music track at low volume into the video."""
    if not os.path.isfile(video_path) or not os.path.isfile(bgm_path):
        return False
    # Loop BGM if needed, keep it at low volume
    args = [
        "-y",
        "-i", video_path,
        "-stream_loop", "-1",
        "-i", bgm_path,
        "-filter_complex", f"[1:a]volume={bgm_vol}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    ok, err = _run_ffmpeg(args, timeout=300)
    if ok:
        logger.info("BGM mixed: %s -> %s", bgm_path, output_path)
    return ok



def _combine_scene_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> tuple[bool, float]:
    """Combine a scene's concatenated video + audio (NO slow-motion).

    Scene-level composition strategy (replaces shot-level _combine_scene_clip):
    - If audio <= video: overlay audio, pad with silence to video duration
    - If audio > video: LOOP video at normal speed to match audio (NO slow-mo!)
    - No audio: output video with silent audio track

    Returns (success, actual_duration_s).
    """
    if not video_path or not os.path.isfile(video_path):
        logger.warning("Scene combine: video file missing: %s", video_path)
        return False, 0.0

    video_dur = _probe_video_duration(video_path)
    if video_dur < 0.5:
        video_dur = 5.0

    audio_dur = 0.0
    if audio_path and os.path.isfile(audio_path):
        audio_dur = _probe_audio_duration(audio_path)

    if audio_path and os.path.isfile(audio_path) and audio_dur > 0.5:
        if audio_dur <= video_dur * 1.02:
            # Audio fits within video: overlay audio, pad with silence to video
            out_dur = video_dur
            dur_str = f"{out_dur:.2f}"
            logger.info(
                "Scene combine: audio %.1fs <= video %.1fs: overlay + silence pad",
                audio_dur, video_dur,
            )
            args = [
                "-y", "-i", video_path, "-i", audio_path,
                "-filter_complex",
                f"[1:a]atrim=0:{dur_str},apad=whole_dur={dur_str}[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                "-t", dur_str, "-r", "30", "-movflags", "+faststart",
                output_path,
            ]
        else:
            # Audio longer than video: LOOP video at NORMAL speed (no slow-mo!)
            out_dur = audio_dur
            dur_str = f"{out_dur:.2f}"
            # stream_loop = enough additional loops to cover audio duration
            stream_loop = int(audio_dur / video_dur)
            logger.info(
                "Scene combine: audio %.1fs > video %.1fs: loop video %dx (normal speed), out=%.1fs",
                audio_dur, video_dur, stream_loop + 1, out_dur,
            )
            input_args = ["-y"]
            if stream_loop > 0:
                input_args += ["-stream_loop", str(stream_loop)]
            input_args += ["-i", video_path, "-i", audio_path]
            args = input_args + [
                "-filter_complex",
                f"[1:a]atrim=0:{dur_str},apad=whole_dur={dur_str}[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                "-t", dur_str, "-r", "30", "-movflags", "+faststart",
                output_path,
            ]
    else:
        # No audio: output video with silent audio track
        out_dur = video_dur
        dur_str = f"{out_dur:.2f}"
        args = [
            "-y", "-i", video_path,
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            "-t", dur_str, "-r", "30", "-movflags", "+faststart",
            output_path,
        ]

    ok, err = _run_ffmpeg(args, timeout=180)
    if ok:
        actual = _probe_video_duration(output_path)
        logger.debug("Scene clip composed: %s (dur=%.2fs)", output_path, actual)
        return True, actual if actual > 0.5 else out_dur
    logger.warning("Scene clip compose failed: %s", err[:200])
    return False, out_dur


async def compose_episode(
    scene_groups: list[dict],
    subtitles: list[dict],
    episode_id: str = "ep_001",
    output_dir: Optional[str] = None,
    bgm_path: str = "",
) -> dict:
    """Compose final episode video with SCENE-LEVEL composition.

    Each scene group:
        {
            "scene_id": int,
            "videos": [{"local_path": str, "duration_s": float, "shot_id": int}],  # ordered
            "audios": [{"local_path": str, "duration_s": float, "shot_id": int, "text": str}],  # ordered
            "shots_meta": [{"shot_id": int, "duration_s": float}],  # optional
        }

    Seed Audio 1.0 生成的音频已包含背景音/BGM/环境音，TTS 音频直接按
    scene 分组拼接后对齐到 scene video，无需 audio_timeline 时间线编排。

    Strategy:
      1. Per scene: concat all shot videos -> one long scene video
      2. Per scene: concat all TTS audios -> scene audio
      3. Overlay scene audio on scene video (NO slow-motion; loop if audio longer)
      4. Concat all scene clips -> final video
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp_compose"
    tmp_dir.mkdir(exist_ok=True)

    # Filter to valid scenes (at least one video file)
    valid_scenes: list[dict] = []
    for sg in scene_groups:
        videos = [v for v in sg.get("videos", []) if v.get("local_path") and os.path.isfile(v["local_path"])]
        if not videos:
            logger.warning("Scene %s: no valid videos, skipping", sg.get("scene_id"))
            continue
        audios = [a for a in sg.get("audios", []) if a.get("local_path") and os.path.isfile(a["local_path"])]
        valid_scenes.append({
            "scene_id": sg.get("scene_id", 0),
            "videos": videos,
            "audios": audios,
            "shots_meta": sg.get("shots_meta", []),
        })

    if not valid_scenes:
        logger.error("No valid scenes to compose")
        return {"final_video_path": "", "success": False, "error": "no valid scenes"}

    logger.info(
        "Composing episode %s (scene-level): %d scenes, %d subtitles",
        episode_id, len(valid_scenes), len(subtitles),
    )

    scene_clips: list[str] = []
    scene_durations: list[float] = []
    scene_shot_offsets: list[dict[int, float]] = []

    for si, sg in enumerate(valid_scenes):
        videos = sg["videos"]
        audios = sg["audios"]

        # Step 1: Concat all shot videos in this scene -> one long video
        if len(videos) == 1:
            scene_video = videos[0]["local_path"]
        else:
            scene_video = str(tmp_dir / f"scene_{si:03d}_video.mp4")
            video_paths = [v["local_path"] for v in videos]
            ok = _concatenate_clips(video_paths, scene_video)
            if not ok or not os.path.isfile(scene_video):
                logger.warning("Scene %d video concat failed, using first shot", si)
                scene_video = videos[0]["local_path"]

        # Step 2: Build scene audio — concat all TTS audios in this scene (in shot order)
        audio_paths = [a["local_path"] for a in audios]
        scene_audio = _concat_audio_list(audio_paths, tmp_dir, si) if len(audio_paths) > 1 else (audio_paths[0] if audio_paths else "")

        # Step 3: Combine scene video + scene audio (NO slow-motion!)
        clip_path = str(tmp_dir / f"scene_{si:03d}.mp4")
        ok, actual_dur = _combine_scene_video_audio(scene_video, scene_audio, clip_path)
        if ok:
            scene_clips.append(clip_path)
            scene_durations.append(actual_dur)
        else:
            logger.warning("Scene %d combine failed, using raw video", si)
            scene_clips.append(scene_video)
            d = _probe_video_duration(scene_video)
            scene_durations.append(d if d > 0.5 else 5.0)

        # Compute shot offsets within scene (for subtitle timing)
        offsets: dict[int, float] = {}
        cum = 0.0
        for v in videos:
            sid = v.get("shot_id")
            if sid is not None:
                offsets[sid] = cum
            cum += v.get("duration_s", 5.0)
        scene_shot_offsets.append(offsets)
        logger.info(
            "Scene %d: %d shots -> %.1fs video, audios -> %.1fs clip",
            si, len(videos), sum(v.get("duration_s", 5.0) for v in videos),
            actual_dur,
        )

    # Step 4: Concat all scene clips -> final video
    concat_path = str(tmp_dir / f"{episode_id}_concat.mp4")
    concat_ok = _concatenate_clips(scene_clips, concat_path)
    if not concat_ok:
        logger.error("Final concatenation failed")
        return {"final_video_path": "", "success": False, "error": "concatenation failed"}

    # Step 5: Use concat video directly as final (no subtitle burning)
    final_path = str(out_dir / f"{episode_id}_final.mp4")

    # Step 6: Mix BGM at low volume
    if bgm_path and os.path.isfile(bgm_path):
        bgm_ok = _mix_bgm(concat_path, bgm_path, final_path)
        if not bgm_ok:
            import shutil
            shutil.copy2(concat_path, final_path)
    else:
        import shutil
        shutil.copy2(concat_path, final_path)

    # Cleanup temp files
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    if os.path.isfile(final_path):
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        total_dur = sum(scene_durations)
        logger.info(
            "Episode composed (scene-level): %s (%.1f MB, ~%.0fs)",
            final_path, size_mb, total_dur,
        )
        return {
            "final_video_path": final_path,
            "success": True,
            "error": "",
            "duration_s": total_dur,
            "size_mb": round(size_mb, 1),
        }

    return {"final_video_path": "", "success": False, "error": "final file not created"}


def compose_episode_sync(
    scene_groups: list[dict],
    subtitles: list[dict],
    episode_id: str = "ep_001",
    output_dir: Optional[str] = None,
    bgm_path: str = "",
) -> dict:
    """Synchronous wrapper for compose_episode."""
    return asyncio.run(compose_episode(
        scene_groups, subtitles, episode_id, output_dir, bgm_path,
    ))
