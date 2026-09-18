# Failure Modes

The system is designed so that expected failures are explicit and testable.
This document lists the failure modes that matter and how each is handled.

## Domain-level

| Failure mode                                | Handling                                             |
|---------------------------------------------|------------------------------------------------------|
| Empty id, negative time, bad weight         | ValueError in __post_init__, covered by tests        |
| Duplicate resource or job id                | ValueError in Problem.__post_init__                  |
| Unknown predecessor                         | ValueError in Problem.__post_init__                  |
| Precedence cycle                            | ValueError in repair._topological_job_order          |
| Invalid dynamic change (unknown id, etc.)   | ValueError in apply_change                           |
| Removing a job that is a predecessor        | ValueError in apply_change                           |

## Algorithm-level

| Failure mode                                | Handling                                             |
|---------------------------------------------|------------------------------------------------------|
| Infeasible candidate                        | Penalty model; Evaluation.feasible == False          |
| Release-time violation in GA child          | repair_release_times                                 |
| Precedence violation in GA child            | repair_precedence (topological, monotone)            |
| Capacity violation                          | Not repaired; handled by penalty (by design)         |
| Stagnation                                  | TerminationPolicy(stagnation_window=...)             |
| Premature convergence                       | Tracked via population_diversity and history         |
| Chromosome length mismatch after a change   | Refused unless adapt_state is called first           |

## Transport-level

| Failure mode                                | Handling                                             |
|---------------------------------------------|------------------------------------------------------|
| Duplicate event on the bus                  | Dropped per subscriber (InMemoryEventBus._seen)      |
| Duplicate Kafka message                     | Dropped per group (KafkaEventConsumer._seen)         |
| Malformed JSON payload                      | ValueError in decode_event; wrapped as ConsumerError |
| Producer send failure                       | Retried per RetryPolicy; final failure raises        |
|                                             | ProducerError                                        |
| Consumer closed mid-poll                    | ConsumerError                                        |
| Kafka not installed                         | RuntimeError from factory._import_kafka              |

## Worker-level

| Failure mode                                | Handling                                             |
|---------------------------------------------|------------------------------------------------------|
| Evaluation raised                           | Captured; FitnessResult.error is set                 |
| Timeout                                     | Job is retried up to max_retries times               |
| Retries exhausted                           | FitnessResult.error set, total == inf                |
| Unpicklable payload                         | Prevented; domain types implement __reduce__         |

## Experiment-level

| Failure mode                                | Handling                                             |
|---------------------------------------------|------------------------------------------------------|
| Duplicate run_id in RunHistory              | ValueError                                           |
| Naive timestamp in RunRecord                | ValueError                                           |
| finished_at < started_at                    | ValueError                                           |
| Missing pyarrow for Parquet export          | RuntimeError with an install hint                    |
| Empty input to summarize                    | ValueError                                           |
| Mismatched lengths in stability_report      | ValueError                                           |

## Operational non-goals

- No claim of real-time behavior: latency is measured by
  sequential_vs_parallel and is not asserted.
- No claim of distributed fault tolerance: the worker model is local.
- No silent recovery: failures propagate explicitly or are captured as
  data, never suppressed.
