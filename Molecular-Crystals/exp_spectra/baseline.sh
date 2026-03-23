#!/bin/bash

csv_dir="./"   # folder with your CSV files
y_col=2          # column number for Y values (1-based)

for file in "$csv_dir"/*.csv; do
    [ -f "$file" ] || continue  # skip if no files

    # Find min value in Y column (ignoring header)
    min_val=$(awk -F, -v col="$y_col" 'NR>1 {if(min=="" || $col<min) min=$col} END{print min}' "$file")
    
    # Subtract min_val from Y column and overwrite same file
    awk -F, -v col="$y_col" -v min="$min_val" 'BEGIN{OFS=","} 
        NR==1 {print; next} 
        {$col=$col-min; print}' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    
    echo "Updated $file (baseline shifted by $min_val)"
done

