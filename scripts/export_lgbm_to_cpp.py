"""Export trained LightGBM model to high-performance standalone C++ tree evaluation code."""
import json
import os
import joblib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
NATIVE_DIR = os.path.join(REPO_ROOT, "src", "native")
os.makedirs(NATIVE_DIR, exist_ok=True)

def tree_to_cpp(node):
    if "leaf_value" in node:
        return f"{node['leaf_value']:.8f}f"
    
    f_idx = node["split_feature"]
    thresh = node["threshold"]
    left_str = tree_to_cpp(node["left_child"])
    right_str = tree_to_cpp(node["right_child"])
    return f"(f[{f_idx}] <= {thresh:.8f}f ? {left_str} : {right_str})"

def generate_cpp_header():
    model_path = os.path.join(MODELS_DIR, "lgbm_matcher.joblib")
    model = joblib.load(model_path)
    dump = model.booster_.dump_model()
    
    trees = dump["tree_info"]
    print(f"Exporting {len(trees)} trees to C++...")
    
    lines = [
        "// Generated LightGBM Native Tree Predictor",
        "#pragma once",
        "#include <cmath>",
        "#include <vector>",
        "",
        "inline float sigmoid(float x) {",
        "    return 1.0f / (1.0f + std::exp(-x));",
        "}",
        "",
        "inline float predict_lgbm_prob(const float* f) {",
        "    float score = 0.0f;",
    ]
    
    for i, t in enumerate(trees):
        tree_expr = tree_to_cpp(t["tree_structure"])
        lines.append(f"    score += {tree_expr};")
        
    lines.append("    return sigmoid(score);")
    lines.append("}")
    lines.append("")
    
    out_path = os.path.join(NATIVE_DIR, "lgbm_model.h")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Header generated: {out_path} ({os.path.getsize(out_path):,} bytes)")

if __name__ == "__main__":
    generate_cpp_header()
