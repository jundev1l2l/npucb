# UCB Algorithm with Neural Processes (NP-UCB) for Contextual Bandit Benchmarks


## How to run code

- For **training** with Contextual Bandit Benchmark datasets, use configuration files in `script/train`
  - ex) Train BANP on Wheel Bandit Dataset
  `python main.py --project train_banp_wheel --default_config script/train/example/default_config.yaml --exp_config script/train/example/exp_config/MODEL=BANP.yaml`
- For **evaluation**, use configuration files in `script/eval`
  - ex) Evaluation on Wheel Bandit with BANP and NP-UCB
  `python main.py --project eval_banp_wheel --default_config script/eval/example/default_config.yaml --exp_config script/eval/example/exp_config/MODEL=BANP_ALGO=NPUCB.yaml`


## Features Implemented

- E-greedy
- UCB
- Thompson Sampling
- LinUCB
- **NP-UCB (Neural Processes + Upper Confidence Bound)**
- **NP-TS (Neural Processes + Thompson Sampling)**


## Experiments

**Dataset (Wheel Bandit)**
  - Context X = `3-dim` features = 2-dim user features (concat) 1-dim arm index
    - unit circle 내에서 랜덤하게 추출된 x,y 위치 + 팔의 인덱스
  - Context Y = `1-dim` reward
    - Normal 분포로부터 랜덤하게 추출된 보상
    - 주어진 컨텍스트에 따라 평균이 달라짐
  - 디테일한 보상 생성 메커니즘은 [Deep Bayesian Bandits Showdown (Riquelme et. al.)](https://github.com/andrewk1/pytorch-deep-bayesian-bandits) 참고

**Training**
  - Train Data / Validation Data : i.i.d. randomly generated
  - 50 ~ 500 Contexts, 10 ~ 100 Targets
  - Learning rate 5.0e-4
  - 100 epochs is enough
      
**Experiment Log**
- Evaluate with Running Average Attention                                                                 `2022-04-06`
- Train & Evaluate NP Models on Mushroom Bandit Data                                                      `2022-04-07`
- Train & Evaluate  NP Models on Stock Bandit Data                                                        `2022-04-07`
- Train NP Models on Statlog / Jester / Covertype / Adult Bandit Data                                     `2022-04-13`
- Evaluate NP Models on Statlog / Jester / Covertype / Adult Bandit Data                                  `2022-04-14`
- Train NP Models on Census Bandit Data                                                                   `2022-04-15`
- Evaluate NP Models on Census Bandit Data                                                                `2022-04-15`
- Train & Evaluate NP Models on Wheel Bandit Data                                                         `2022-04-16`
- Evaluate NP-UCB/TS-Online on Wheel Bandit Data                                                       `2022-04-26`
- Search Params for NP-UCB-Online on Wheel Bandit Data                                                    `2022-04-30`
- Evaluate NP-UCB/TS-Online on Mushroom / Stock / Statlog / Jester / Covertype / Adult Bandit Data     `2022-05-20`
- Train Autoencoder Feature Extractor on Bandit Data                                                      `Stopped`
- Train NPs on GP Data (3d, 10d, 30d)                                                                     `Stopped`
- Evaluate NP Bandits Pretrained on GP + Autoencoder Feature Extractor on Wheel                           `Stopped`

**Development Log**
- Training NP Models with Bandit Data                                                                     `2022-03`
- UCB with NPs (NP-UCB)                                                                                   `2022-03`
- ThopmsonSampling with NPs (NP-TS)                                                                       `2022-04-01`
- Refactoring (Add Worker Module, Merge Train / Eval Codes)                                               `2022-04-03`
- Bandit Envs from [Deep Bayeian Bandits Showdown] (Mushroom)                                             `2022-04-05`
- Bandit Envs from [Deep Bayeian Bandits Showdown] (Stock)                                                `2022-04-07`
- Bandit Envs from [Deep Bayeian Bandits Showdown] (Statlog, Jester, Covertype, Adult, Census)            `2022-04-12`
- Bandit Envs from [Deep Bayeian Bandits Showdown] (Wheel)                                                `2022-04-15`
- Online Bandit Algorithms with NPs (NP-UCB/TS-Online)                                                 `2022-04-25`
- Feature Extractor (ex. Autoencoder) For Bandit Data                                                     `2022-04-28`
- Data Sampler, Dataset For GP Training                                                                   `2022-04-29`
- Evaluation Logic using Feature Extractor                                                                `2022-04-30`
- Evaluation Logic with Changing Arm Pools / Envs                                                         `2022-05-11`
- Training Logic with Mixed Envs (Wheel Bandit)                                                           `2022-05-14`


## Dataset (Bandit Environment)

### [Deep Bayesian Bandits Showdown (Riquelme et. al.)](https://github.com/andrewk1/pytorch-deep-bayesian-bandits)
The repo contains 7 contextual bandit simulation data (wheel, jester, mushroom, statlog, adult, covertype, census) so that we can reproduce the bandit environment reported in their paper.


## References

- `andrewk1/pytorch-deep-bayesian-bandits` https://github.com/andrewk1/pytorch-deep-bayesian-bandits
- Deep Bayesian Bandits Showdown (ICLR '18) https://research.google/pubs/pub46647/
