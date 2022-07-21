# UCB Algorithm with Neural Processes (NP-UCB) for News Recommendation


## How to run code

- For **training** with Yahoo R6a Dataset, use configuration files in `script/train`
  - ex) Train BANP on Day 1 data
  ```
  python main.py \
  --project train_banp_news \
  --default_config script/train/example/default_config.yaml \
  --exp_config script/train/example/exp_config/MODEL=BANP.yaml
  ```
- For **evaluation**, use configuration files in `script/eval`
  - ex) Evaluation on Day 1 with BANP and NP-UCB
  ```
  python main.py \
  --project eval_banp_news \
  --default_config script/eval/example/default_config.yaml \
  --exp_config script/eval/example/exp_config/MODEL=BANP_ALGO=NPUCB.yaml
  ```


## Features Implemented

- E-greedy
- UCB
- Thompson Sampling
- LinUCB
- **NP-UCB (Neural Processes + Upper Confidence Bound)**
- **NP-TS (Neural Processes + Thompson Sampling)**


## Experiments

**Dataset**
  - Context X = `12-dim` features = 6-dim user features (concat) 6-dim article features
    - [0, 1] 사이, 평균 0.33, 표준편차 0.38 로 평준화되어 있음
  - Context Y = `1-dim` reward (0 or 1)
    - reward 는 user 에게 추천한 arm (article) 을 클릭했는지(1) 안 했는지(0)
    - 평균 클릭 비율이 3.4% 로 매우 낮음. 즉 거의 모든 Y 값이 0 임
      - Reward Balance 실험: reward 가 1 인 event 의 비율을 0.1/0.3/0.5 로 맞춰서 학습

**Training**
  - Train Data : Day 1 (args.day_train=1)
  - Validation Data : Day 2 (args.day_val=1)
  - 50 Context, 10 Target per Data (args.n_ctx=50, args.n_tar=10)
    - More ctx/tar 실험: ctx 개수를 최대 10/50/100/500/1000 개, tar 개수를 10/50 개로 늘려서 학습 
  - 128 Data per Minibatch (args.batch_size=128)
    - GPU 메모리를 고려하여, Batch Size 는 ctx/tar 개수에 맞춰서 설정 
    - ctx 10/50/100 까지는 128, 500 은 8, 1000 은 2 로 설정 (aigs 서버 기준)

**Experiment Log**
- Training, NLL loss                         
- Training, NLL loss, More ctx/tar           
- Increase Model Capacity                   
- Dataset Reward Balance                    
- Running Average Attention            
- NP-UCB Bandit Evaluation
- NP-TS Bandit Evaluation
- Training, Random Number Ctx/Tar
- Training, Mushroom Bandit Data

**Development Log**
- Meta-training of NPs                                                `2022-02-26`
- Loss Clip                                                           `2022-02-27`
- L2 loss                                                             `2022-02-27`
- Increase Model Capacity                                             `2022-03-01`
- Dataset Reward Balance                                              `2022-03-01`
- Simple Supervised Learning (+ MLP)                                  `2022-03-02`
- Refactoring (Module, Config, Wandb)                                 `2022-03-24`
- ThopmsonSampling with NPs (NP-TS)                                   `2022-04-01`
- Refactoring (Add Worker Module, Merge Train / Eval Codes)           `2022-04-03`
- Bandit Envs from [Deep Bayeian Bandits Showdown] (Mushroom)         `2022-04-05`


## Dataset

### R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 (1.1 GB)
The dataset contains 45,811,883 user visits to the Today Module. For each visit, both the user and each of the candidate articles are associated with a feature vector of dimension 6 (including a constant feature), constructed using a conjoint analysis with a bilinear model.
The dataset can be found [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r).


## References

- `astonismand/Personalized-News-Recommendation` https://github.com/antonismand/Personalized-News-Recommendation
- A Contextual-Bandit Approach to Personalized News Article Recommendation https://arxiv.org/pdf/1003.0146.pdf
- Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms  https://arxiv.org/pdf/1003.5956.pdf
    Used algorithm 2 as a policy evaluator (for finite data stream)
