import os
import pandas as pd
import json
import time
from ..demo.catch_aws_client import AWSClient

def get_iteration_train_data(video_num, child_id):
    score = 0
    appeared_actions = {}
    action_logs_path = f'처리_결과/{video_num}/action.log'

    with open(action_logs_path, 'r') as f:
        action_logs = json.load(f)

    for timestamp, info in action_logs.items():
        if child_id in info:
            action_log = info[child_id]

            for action_with_conf in action_log['actions']:
                action = action_with_conf['action']

                if action in appeared_actions:
                    score += 1/appeared_actions[action]

                appeared_actions[action] = 0

            for action in appeared_actions.keys():
                appeared_actions[action] += 1

    return score


if __name__ == '__main__':
    df = pd.DataFrame()

    bucket_name = input('버킷 이름을 입력해주세요:')
    aws_client = AWSClient(bucket_name)

    while True:
        print('처리 진행을 원하시면 1을, 프로그램 종료를 원하시면 q를 눌러주세요:', end='')
        next = input()
        if next == 'q':
            break

        print('영상 번호를 입력해주세요:', end='')
        video_num = int(input())

        print('해당 영상 아이 객체의 id를 입력해주세요:', end='')
        child_id = input()

        print('처리중입니다...')
        start = time.time()
        score = get_iteration_train_data(video_num, child_id)
        df = df.append(pd.DataFrame(data=[[video_num, score]], columns=['video_num', 'score']))
        print(f'처리 완료. 걸린 시간: {round(time.time()-start, 1)}초')
    
    if not df.empty:
        print(f'데이터를 저장중입니다...')
        df.to_csv(f'지표-반복하기-학습데이터.csv', index=False)

    print(f'프로그램이 정상적으로 종료되었습니다.')