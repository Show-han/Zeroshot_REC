import argparse
import threading
import os
import json
import dotenv
from gentopia import chat
from gentopia.assembler.agent_assembler import AgentAssembler
from gentopia.output import enable_log
from tqdm import tqdm
import time
import logging

lock = threading.Lock()
semaphore = threading.Semaphore(10)

def arg_parser():
    parser = argparse.ArgumentParser(description='Assemble an agent with given name.')
    parser.add_argument('--fewshot_num', type=int, default=45, help='Number of the fewshot examples.')
    parser.add_argument('--fewshot_caption_dir', type=str, default='fewshot.txt', help='Location of the fewshot example captions.')
    parser.add_argument('--fewshot_triplets_dir', type=str, default='fewshot.json', help='Location of the fewshot example caption groundtruth.')
    parser.add_argument('input_dir', type=str, help='Input directory or file.')
    parser.add_argument('output_dir', type=str, help='Output derectory or file.')
    
    args = parser.parse_args()
    return args


class TokenBucket:
    def __init__(self, tokens, refill_rate):
        self.tokens = tokens
        self.capacity = tokens
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def take(self, count=1):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill

            # refill tokens based on the time passed
            self.tokens += elapsed * self.refill_rate
            self.tokens = min(self.tokens, self.capacity)
            self.last_refill = now

            if self.tokens >= count:
                self.tokens -= count
                return True
            return False

    def wait_for_token(self, count=1):
        while not self.take(count):
            sleep_time = max(1, (count - self.tokens) / self.refill_rate)
            time.sleep(sleep_time)


bucket = TokenBucket(tokens=85000, refill_rate=1250) 

def _get_few_shots(number=5, caption='fewshot.txt', triplets='fewshot.json'):
    
    with open(caption, 'r') as f:
        captions = f.read().splitlines()
    
    with open(triplets, 'r') as f:
        triplets_list = json.load(f)

    if len(captions) != len(triplets_list):
        raise ValueError("The number of lines in captions.txt and triplets.json must be the same.")
    
    few_shots = []
    for i in range(number):
        sentence = f'{captions[i]}\n'
        few_shots.append("INPUT: " + sentence + "OUTPUT: " + json.dumps(triplets_list[i], ensure_ascii=False))
    
    return few_shots


def get_and_save_single_triplet(agent, agent_4, input, save_dir):
    attempts = 2
    total_attempts = 4

    while total_attempts > 0:
        try:
            if attempts > 0:
                response = agent.run(input['input'])
            else:
                response = agent_4.run(input['input'])

            output_data = json.loads(response.output)
            triplets_exist = len(output_data.get('relations', [])) > 0
            all_triplets_valid = all(len(triplet) == 3 for triplet in output_data.get('relations', []))
            entity_present = any(output_data['entity'] in (triplet[0], triplet[2]) for triplet in output_data.get('relations', []))
            if not triplets_exist or not all_triplets_valid or not entity_present:
                raise ValueError("Invalid entity or triplet structure.")
            else:
                break

        except Exception as e:
            total_attempts -= 1
            attempts -= 1
            if total_attempts <= 0:
                print(f"Failed to decode response for input: {input['input']}")
                output_data = {"entity": f"{input['input']}", "relations": [[f"{input['input']}", "", ""]]}
                break
            else:
                continue 

    data_to_save = {
        input['id']: output_data
    }

    with open(save_dir, 'a') as f:
        f.write(json.dumps(data_to_save, ensure_ascii=False) + "\n")

    return


def threaded_triplet(agent, agent_4, input, save_dir):
    global bucket

    estimated_token_usage = 2200
    bucket.wait_for_token(estimated_token_usage)

    get_and_save_single_triplet(agent, agent_4, input, save_dir)


    semaphore.release()


def get_triplets_batch(args):
    agent_config_path = f'./chatgpt.yaml'
    agent_config_path_4 = f'./gpt4.yaml'
    assembler = AgentAssembler(file=agent_config_path)
    assembler_4 = AgentAssembler(file=agent_config_path_4)
    agent = assembler.get_agent()
    agent_4 = assembler_4.get_agent()

    with open(args.input_dir, 'r') as f:
        jsonl_file = [json.loads(line) for line in f]

    data = []
    for image in jsonl_file:
        file_name = "_".join(image["file_name"].split("_")[:-1])+".jpg"
        sentences = image["sentences"]
        for s in sentences:
            sent_id = s["sent_id"]
            sentence = s["raw"].lower()
            data_to_save = {
                'id': f"{file_name}_{sent_id}",
                'input': f'{sentence}'
            }
            data.append(data_to_save)

    agent.examples = _get_few_shots(number=args.fewshot_num, caption=args.fewshot_caption_dir, triplets=args.fewshot_triplets_dir)
    agent_4.examples = _get_few_shots(number=args.fewshot_num, caption=args.fewshot_caption_dir, triplets=args.fewshot_triplets_dir)
    # print(agent.examples)

    threads = []

    for input in tqdm(data, total=len(data)):
        semaphore.acquire()
        t = threading.Thread(target=threaded_triplet, args=(agent, agent_4, input, args.output_dir))
        threads.append(t)
        t.start()

    for t in threads:
        t.join() 

    return


if __name__ == '__main__':
    enable_log(log_level='info')
    dotenv.load_dotenv(".env")
    args = arg_parser()
    get_triplets_batch(args)