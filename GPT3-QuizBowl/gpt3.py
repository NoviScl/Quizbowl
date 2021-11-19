import os
import openai

openai.api_key = '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'
openai.Engine.retrieve('davinci-msft')

def date_time_prompt(query: str):
  prompt = ''
  prompt += 'Expression: end time - start time\n'
  prompt += 'Header: time duration\n\n'
  prompt += 'Expression: end term - start term\n'
  prompt += 'Header: term duration\n\n'
  prompt += 'Expression: left office - took office\n'
  prompt += 'Header: time in office\n\n'
  prompt += f'Expression: {query}\n'
  prompt += 'Header:'
  return prompt

aa = openai.Completion.create(
  engine='davinci-msft',
  prompt=date_time_prompt('until - from'),
  temperature=0.0,
  max_tokens=5,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=['\n']
)

print(aa['choices'][0]['text'])


# papp_cloud login -u zy-z19@mails.tsinghua.edu.cn -p <<EOF
# 3df119ab7
# EOF
# papp_cloud ssh scv0540@bscc-n22


