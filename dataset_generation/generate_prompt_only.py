import json

def process_jsonl(input_file, output_file_0, output_file_1):
    with open(input_file, 'r') as infile:
        with open(output_file_0, 'w') as file_0, open(output_file_1, 'w') as file_1:
            for line in infile:
                record = json.loads(line)
                prompt = record['prompt']
                safety = record['safety']
                if safety == 0:
                    file_0.write(prompt + '\n')
                elif safety == 1:
                    file_1.write(prompt + '\n')

# Usage
input_file = 'Base_accept_reject.jsonl'
output_file_0 = 'harmful.txt'
output_file_1 = 'benign.txt'
process_jsonl(input_file, output_file_0, output_file_1)
