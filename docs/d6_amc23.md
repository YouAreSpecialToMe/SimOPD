# D6: size vs capability (amc23)

Student Qwen3-1.7B-Base, acc 0.1820, 40 problems.
`T>S` is the fraction of problems the teacher gets right and the student
does not -- the supervision that actually exists. `S>T` is the reverse,
where distilling teaches the wrong answer; no ceiling shows it.

| teacher | size x | acc | T>S | S>T | headroom |
|---|---|---|---|---|---|
| Qwen3-1.7B | 1.0 | 0.4336 | 82.50% | 5.00% | 0.2516 |
| Qwen3-4B-Instruct-2507 | 2.4 | 0.9133 | 100.00% | 0.00% | 0.7313 |
| Qwen3-8B | 4.7 | 0.6828 | 95.00% | 0.00% | 0.5008 |
