prompt="This 17th-century English philosopher who wrote Novum Organum and spearheaded the Scientific Revolution. His name is "

# echo "${prompt}"
# curl https://api.openai.com/v1/engines/davinci-msft \
curl https://api.openai.com/v1/engines/davinci-msft/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer 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' \
  -d '{
  "prompt": $prompt,
  "max_tokens": 5
}'

chenglei_key=sk-UCngIaUfTZc5em4h0eqkMax1orLcjc9qzMdMftfD
msr_key=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

# ## retriver available engines
# curl https://api.openai.com/v1/engines \
#   -H 'Authorization: Bearer sk-UCngIaUfTZc5em4h0eqkMax1orLcjc9qzMdMftfD'
