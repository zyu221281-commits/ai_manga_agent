"""跨集三视图一致性链路功能测试（无需 DB / API）。

验证：
1. register_multi_view_anchor + get_canonical_appearance 存取
2. 首集写入后后续集不覆盖（no drift）
3. _build_video_prompt 注入 canonical appearance
4. _build_canonical_appearance 从 id_card 构建
"""
import sys
sys.path.insert(0, '.')


def main():
    # 测试1: register + get_canonical_appearance
    # 使用模块级单例（与 composer._build_video_prompt / generate_multi_view_anchors 一致）
    from app.quality.character_consistency import character_consistency as checker
    checker.clear()  # 清空避免重复运行污染

    appearance = (
        "Lin Feng, black hair, amber eyes, wearing dark robe, slim body, "
        "scar on left eye, consistent character appearance, same person throughout"
    )
    checker.register_multi_view_anchor(
        name="Lin Feng",
        views={"front": "/tmp/front.png", "side": "/tmp/side.png", "back": "/tmp/back.png"},
        image_urls={"front": "http://example.com/front.png"},
        appearance_text=appearance,
    )
    got = checker.get_canonical_appearance("Lin Feng")
    assert got == appearance, f"canonical appearance mismatch: {got!r}"
    print("[OK] test1: get_canonical_appearance returns seed_prompt")

    # 测试2: has_multi_view
    assert checker.has_multi_view("Lin Feng")
    print("[OK] test2: has_multi_view=True")

    # 测试3: get_best_ref_view 按 camera_angle 选视角
    # side-angle → side 视图：side 本地路径存在 → 返回本地路径
    ref = checker.get_best_ref_view("Lin Feng", "side-angle")
    assert ref == "/tmp/side.png", f"unexpected ref: {ref}"
    # close-up → front 视图：front URL 存在 → 优先返回 URL
    ref2 = checker.get_best_ref_view("Lin Feng", "close-up")
    assert ref2 == "http://example.com/front.png", f"unexpected ref2: {ref2}"
    # back → back 视图：back 无 URL，有本地路径 → 返回本地路径
    ref3 = checker.get_best_ref_view("Lin Feng", "back")
    assert ref3 == "/tmp/back.png", f"unexpected ref3: {ref3}"
    print("[OK] test3: get_best_ref_view by camera_angle (front=URL, side=back local, back=local)")

    # 测试4: 首集写入后，后续集不应覆盖 canonical appearance（防 drift）
    drifted = "Lin Feng, blonde hair, blue eyes (drifted)"
    checker.register_multi_view_anchor(
        name="Lin Feng", views={}, image_urls={}, appearance_text=drifted,
    )
    got2 = checker.get_canonical_appearance("Lin Feng")
    assert got2 == appearance, f"should NOT be overwritten: {got2!r}"
    print("[OK] test4: canonical appearance not overwritten by later episodes (no drift)")

    # 测试5: _build_video_prompt 注入 canonical appearance
    from app.agents.composer import ComposerAgent
    agent = ComposerAgent(session=None, episode_id="test")

    cs = {
        "shot": {"description": "hero draws sword"},
        "scene_data": {"emotion": "tense"},
        "character_name": "Lin Feng",
    }
    prompt = agent._build_video_prompt(cs)
    assert "Lin Feng" in prompt, f"name missing: {prompt!r}"
    assert "black hair" in prompt, f"appearance missing: {prompt!r}"
    assert "amber eyes" in prompt, f"eyes missing: {prompt!r}"
    assert "dark robe" in prompt, f"outfit missing: {prompt!r}"
    print("[OK] test5: _build_video_prompt injects canonical appearance")
    print("     prompt =", prompt)

    # 测试6: 无 anchor 时兜底
    cs2 = {
        "shot": {"description": "crowd scene"},
        "scene_data": {"emotion": "calm"},
        "character_name": "UnknownExtra",
    }
    prompt2 = agent._build_video_prompt(cs2)
    assert "consistent character appearance" in prompt2, f"fallback missing: {prompt2!r}"
    print("[OK] test6: _build_video_prompt fallback when no anchor")

    # 测试7: _build_canonical_appearance 从 id_card 构建
    from app.agents.asset_manager import AssetManagerAgent
    id_card = {
        "hair_color": "#8B4513", "eye_color": "amber",
        "outfit": "魏晋青衫", "body_type": "slim",
        "distinguishing_features": "左眼疤痕",
        "negative_traits": "六指",
    }
    ap = AssetManagerAgent._build_canonical_appearance("苏哲", id_card)
    assert "#8B4513 hair" in ap and "amber eyes" in ap
    assert "魏晋青衫" in ap and "左眼疤痕" in ap and "slim body" in ap
    print("[OK] test7: _build_canonical_appearance from id_card")
    print("     appearance =", ap)

    print()
    print("=== ALL TESTS PASSED: 三视图跨集一致性链路完整 ===")


if __name__ == "__main__":
    main()
