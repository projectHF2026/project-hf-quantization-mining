# modified from fromPretrainedParser.py to focus on quantized code models

import sys
import re
import os
from pathlib import Path

# Load your quantized code models list
print("Loading quantized code models list...")
with open('code_quantized_models.txt', 'r') as f:
    QUANTIZED_MODELS = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(QUANTIZED_MODELS)} quantized code models")

# Directory containing grep results
GREP_DIR = 'grepResults_quantized/'

# Get list of result files
if not os.path.exists(GREP_DIR):
    print(f"Error: Directory {GREP_DIR} not found!")
    exit(1)

result_files = [f for f in os.listdir(GREP_DIR) if f.endswith('.txt')]
print(f"Found {len(result_files)} result files to process\n")

if len(result_files) == 0:
    print("No result files to process. Exiting.")
    exit(0)

# Output files
out_prj_modelList = open('prj_quantized_modelList.csv', 'w')
out_prj_file_model = open('prj_file_quantized_model.csv', 'w')
out_matched_models = open('matched_quantized_models.csv', 'w')

# Write headers
out_prj_modelList.write("project,models,model_count\n")
out_prj_file_model.write("project,file_path,model_name,loading_method\n")
out_matched_models.write("model_name,project,file_path,loading_method\n")

# Regex patterns for different loading methods
PATTERNS = {
    'from_pretrained': r'\.from_pretrained\s*\(\s*["\']([^"\']+)["\']',
    'GPTQ': r'(?:AutoGPTQForCausalLM|GPTQModel)\.from_quantized\s*\(\s*["\']([^"\']+)["\']',
    'AWQ': r'AutoAWQForCausalLM\.from_quantized\s*\(\s*["\']([^"\']+)["\']',
    'GGUF_llama': r'Llama\s*\(\s*model_path\s*=\s*["\']([^"\']+\.gguf)["\']',
    'GGUF_ctransformers': r'AutoModelForCausalLM\.from_pretrained\s*\(\s*["\']([^"\']+)["\'].*model_type\s*=\s*["\']llama["\']',
    'config': r'model_name_or_path\s*=\s*["\']([^"\']+)["\']',
    'bitsandbytes': r'BitsAndBytesConfig.*["\']([^"\']+)["\']',
}

total_projects = 0
total_matches = 0
matched_models_set = set()

for result_file in result_files:
    file_path = os.path.join(GREP_DIR, result_file)
    prj_name = result_file.replace('£sep£', '/').replace('.txt', '')
    
    print(f"Processing: {prj_name}")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as ff:
        lines = ff.readlines()
    
    models_found = []
    
    for line in lines:
        # Split on first colon to separate file path from content
        parts = line.split(':', 1)
        if len(parts) < 2:
            continue
        
        file_location = parts[0]
        code_content = parts[1]
        
        # Try each pattern
        for method, pattern in PATTERNS.items():
            matches = re.findall(pattern, code_content, re.IGNORECASE)
            
            for raw_match in matches:
                # Clean up the match
                model_name = raw_match.strip().strip('"\'')
                
                # Remove common prefixes/suffixes
                model_name = model_name.replace('models/', '').replace('.gguf', '')
                
                # Check if this matches any of your quantized models
                # Direct match
                if model_name in QUANTIZED_MODELS:
                    if model_name not in models_found:
                        models_found.append(model_name)
                        matched_models_set.add(model_name)
                        total_matches += 1
                    
                    out_prj_file_model.write(f"{prj_name},{file_location},{model_name},{method}\n")
                    out_matched_models.write(f"{model_name},{prj_name},{file_location},{method}\n")
                    print(f"  ✓ MATCH: {model_name} (method: {method})")
                
                # Partial match (model name contains part of your list)
                else:
                    for quant_model in QUANTIZED_MODELS:
                        # Check if there's overlap between the found name and your list
                        if quant_model.lower() in model_name.lower() or model_name.lower() in quant_model.lower():
                            if quant_model not in models_found:
                                models_found.append(quant_model)
                                matched_models_set.add(quant_model)
                                total_matches += 1
                            
                            out_prj_file_model.write(f"{prj_name},{file_location},{quant_model} (via {model_name}),{method}\n")
                            out_matched_models.write(f"{quant_model},{prj_name},{file_location},{method}\n")
                            print(f"  ✓ PARTIAL MATCH: {quant_model} <- {model_name} (method: {method})")
                            break
        
        # Also check for direct model name mentions (e.g., in comments, configs)
        for quant_model in QUANTIZED_MODELS:
            if quant_model in code_content:
                if quant_model not in models_found:
                    models_found.append(quant_model)
                    matched_models_set.add(quant_model)
                    total_matches += 1
                    
                    out_prj_file_model.write(f"{prj_name},{file_location},{quant_model},direct_mention\n")
                    out_matched_models.write(f"{quant_model},{prj_name},{file_location},direct_mention\n")
                    print(f"  ✓ DIRECT MENTION: {quant_model}")
    
    if models_found:
        total_projects += 1
        out_prj_modelList.write(f"{prj_name},{';'.join(models_found)},{len(models_found)}\n")
        print(f"  → Total models found in this project: {len(models_found)}\n")
    else:
        print(f"  ✗ No matching quantized models found\n")

# Close output files
out_prj_modelList.close()
out_prj_file_model.close()
out_matched_models.close()

# Summary
print("=" * 60)
print("PARSING COMPLETE")
print("=" * 60)
print(f"Total projects with matches: {total_projects}")
print(f"Total model matches: {total_matches}")
print(f"Unique quantized models found: {len(matched_models_set)}")
print(f"Coverage: {len(matched_models_set)}/{len(QUANTIZED_MODELS)} ({100*len(matched_models_set)/len(QUANTIZED_MODELS):.1f}%)")
print("\nOutput files:")
print(f"  - prj_quantized_modelList.csv (projects → models)")
print(f"  - prj_file_quantized_model.csv (detailed file locations)")
print(f"  - matched_quantized_models.csv (models → projects)")
print("=" * 60)

# Print most used models
print("\nTop 10 most used quantized code models:")
model_usage = {}
for model in matched_models_set:
    with open('matched_quantized_models.csv', 'r') as f:
        count = sum(1 for line in f if line.startswith(model + ','))
    model_usage[model] = count

for i, (model, count) in enumerate(sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"{i:2d}. {model:50s} - {count:3d} projects")