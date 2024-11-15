pip freeze > temp_requirements.txt
cat temp_requirements.txt >> requirements.txt
sort -u requirements.txt -o requirements.txt
rm temp_requirements.txt