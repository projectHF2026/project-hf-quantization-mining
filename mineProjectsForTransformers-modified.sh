#!/bin/bash

# Input: CSV from scraper (deduplicated version)
filename='quantized_model_dependents_dedup.csv'

# Directory to store grep results
resultsDir='grepResults_quantized/'
mkdir -p $resultsDir

# GitHub credentials (REQUIRED - replace with your credentials)
username='YOUR_GITHUB_USERNAME'
token='YOUR_GITHUB_TOKEN'

echo "=========================================="
echo "Starting Mining Process"
echo "Input file: $filename"
echo "Results directory: $resultsDir"
echo "=========================================="
echo ""

n=1

# Skip header line
tail -n +2 "$filename" | while IFS=',' read -r repo_path stars forks source_libs; do
    
    echo "=========================================="
    echo "Processing repo #$n"
    echo "Repository: $repo_path"
    echo "Stars: $stars | Forks: $forks"
    echo "Source libraries: $source_libs"
    echo "=========================================="
    
    # Extract owner and repo name
    owner=$(echo "$repo_path" | cut -d "/" -f 2)
    dirName=$(echo "$repo_path" | cut -d "/" -f 3)
    
    # Construct clone URL
    prjPath="https://${username}:${token}@github.com${repo_path}.git"
    
    echo "Cloning: ${repo_path}"
    
    # Clone with depth=1 (faster, only latest commit)
    if git clone --depth=1 "$prjPath" 2>/dev/null; then
        echo "✓ Clone successful"
        
        # Create output filename
        fileGrep="${owner}£sep£${dirName}.txt"
        
        echo "Searching for quantized model usage..."
        
        # Search for:
        # 1. .from_pretrained calls
        # 2. Quantization-specific loaders
        # 3. GGUF file references
        # 4. Model configuration files
        
        grep -RiIE \
            "(\.from_pretrained|AutoGPTQForCausalLM|AutoAWQForCausalLM|\.gguf|quantization_config|Llama\(|GPTQModel|load_quantized|BitsAndBytesConfig|from_quantized)" \
            "./$dirName" > "${resultsDir}${fileGrep}" 2>/dev/null
        
        # Also search specifically for model names from your list
        if [ -f "code_quantized_models.txt" ]; then
            grep -RiIF \
                -f code_quantized_models.txt \
                "./$dirName" >> "${resultsDir}${fileGrep}" 2>/dev/null
        fi
        
        # Check if we found anything
        if [ -s "${resultsDir}${fileGrep}" ]; then
            line_count=$(wc -l < "${resultsDir}${fileGrep}")
            echo "✓ Found potential matches ($line_count lines)"
        else
            echo "✗ No matches found"
            rm "${resultsDir}${fileGrep}"  # Remove empty files
        fi
        
        # Cleanup
        rm -rf "./$dirName"
        echo "✓ Cleaned up cloned repo"
    else
        echo "✗ Clone failed (repo might be private/deleted/moved)"
    fi
    
    # Rate limiting
    remainder=$((n % 100))
    if [ "$remainder" -eq 0 ]; then
        echo ""
        echo "⏸ Rate limiting pause (30 min) after 100 repos..."
        echo "Processed: $n repos"
        sleep 1800
    fi
    
    # Small delay between clones
    sleep 2
    
    n=$((n+1))
    echo ""
done

echo ""
echo "=========================================="
echo "MINING COMPLETE"
echo "=========================================="
echo "Results stored in: $resultsDir"
echo "Total repos processed: $((n-1))"
echo "Total result files: $(ls -1 $resultsDir | wc -l)"
echo "=========================================="