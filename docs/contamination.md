# Decontamination: 13-gram overlap vs the training set

Training set: `/home/zz865/data/simopd_math/train.parquet`, 14476 rows.

Duplicate = a problem sharing >=80% of its 13-grams with training data. A single shared gram is not contamination: competition boilerplate ("can be written as m/n where m and n are relatively prime positive integers") is thirteen tokens by itself.

| benchmark | duplicates (>=80%) | partial (40%-80%) | any shared gram | total | verdict |
|---|---|---|---|---|---|
| math500 | 7 | 1 | 17 | 500 | DUPLICATES |
| amc23 | 0 | 0 | 11 | 40 | CLEAN |
| aime24 | 0 | 0 | 9 | 30 | CLEAN |
| aime25 | 0 | 0 | 9 | 30 | CLEAN |
| minerva | 0 | 0 | 0 | 272 | CLEAN |
