with open("ssef_cam_test.c", "r") as f:
    code = f.read()

# 1. Swap B and R when reading from cam_buf (RGA outputs BGR888)
old_lb = """            drow[dx * 3 + 0] = (int8_t)(srow[sx * 3 + 0] ^ 0x80);
            drow[dx * 3 + 1] = (int8_t)(srow[sx * 3 + 1] ^ 0x80);
            drow[dx * 3 + 2] = (int8_t)(srow[sx * 3 + 2] ^ 0x80);"""

new_lb = """            drow[dx * 3 + 0] = (int8_t)(srow[sx * 3 + 2] ^ 0x80);
            drow[dx * 3 + 1] = (int8_t)(srow[sx * 3 + 1] ^ 0x80);
            drow[dx * 3 + 2] = (int8_t)(srow[sx * 3 + 0] ^ 0x80);"""

old_crop = """                dp[0] = (int8_t)(sp[0] ^ 0x80);
                dp[1] = (int8_t)(sp[1] ^ 0x80);
                dp[2] = (int8_t)(sp[2] ^ 0x80);"""

new_crop = """                dp[0] = (int8_t)(sp[2] ^ 0x80);
                dp[1] = (int8_t)(sp[1] ^ 0x80);
                dp[2] = (int8_t)(sp[0] ^ 0x80);"""

assert old_lb in code and old_crop in code
code = code.replace(old_lb, new_lb).replace(old_crop, new_crop)

# 2. Raise palm score threshold from 0.50 to 0.65 to eliminate 0.50 noise floor
code = code.replace("if (score < 0.50f) continue;", "if (score < 0.65f) continue;")

# 3. Print top-3 logits & raw presence/handedness values for clarity
old_pres = """                float presence = dequant_i8(pres_i8[0], &lmk_m.out_attr[1]);
                float handed   = dequant_i8(hand_i8[0], &lmk_m.out_attr[2]);"""

new_pres = """                float raw_p = dequant_i8(pres_i8[0], &lmk_m.out_attr[1]);
                float raw_h = dequant_i8(hand_i8[0], &lmk_m.out_attr[2]);
                float presence = (raw_p >= 0.0f && raw_p <= 1.0f) ? raw_p : (1.0f / (1.0f + expf(-raw_p)));
                float handed   = (raw_h >= 0.0f && raw_h <= 1.0f) ? raw_h : (1.0f / (1.0f + expf(-raw_h)));"""

assert old_pres in code
code = code.replace(old_pres, new_pres)

with open("ssef_cam_test.c", "w") as f:
    f.write(code)
print("Patched ssef_cam_test.c successfully.")
