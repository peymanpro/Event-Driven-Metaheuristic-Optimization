from __future__ import annotations

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.domain.problem import Problem


def _topological_job_order(problem: Problem) -> list[int]:
    """Return job indices in topological order of the precedence DAG.

    Raises ``ValueError`` if the precedence graph contains a cycle.
    """
    n = len(problem.jobs)
    index_by_id = {job.id: i for i, job in enumerate(problem.jobs)}
    indegree = [0] * n
    successors: list[list[int]] = [[] for _ in range(n)]
    for i, job in enumerate(problem.jobs):
        for pred_id in job.predecessors:
            j = index_by_id[pred_id]
            indegree[i] += 1
            successors[j].append(i)
    queue = [i for i in range(n) if indegree[i] == 0]
    order: list[int] = []
    while queue:
        i = queue.pop()
        order.append(i)
        for j in successors[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)
    if len(order) != n:
        raise ValueError("precedence graph contains a cycle")
    return order


def repair_release_times(problem: Problem, chromosome: Chromosome) -> Chromosome:
    """Raise any start time below the job's release time."""
    starts = list(chromosome.start_time)
    changed = False
    for i, job in enumerate(problem.jobs):
        if starts[i] < job.release_time:
            starts[i] = job.release_time
            changed = True
    if not changed:
        return chromosome
    return Chromosome(resource_idx=chromosome.resource_idx, start_time=tuple(starts))


def repair_precedence(problem: Problem, chromosome: Chromosome) -> Chromosome:
    """Push jobs later so each starts at or after all its predecessors finish.

    Uses a single pass in topological order; start times are only increased.
    """
    order = _topological_job_order(problem)
    index_by_id = {job.id: i for i, job in enumerate(problem.jobs)}
    starts = list(chromosome.start_time)
    changed = False
    for i in order:
        job = problem.jobs[i]
        for pred_id in job.predecessors:
            pi = index_by_id[pred_id]
            pred = problem.jobs[pi]
            pred_resource = problem.resources[chromosome.resource_idx[pi]]
            pred_finish = starts[pi] + pred.processing_time / pred_resource.speed
            if starts[i] < pred_finish:
                starts[i] = pred_finish
                changed = True
        # Release time also respected.
        if starts[i] < job.release_time:
            starts[i] = job.release_time
            changed = True
    if not changed:
        return chromosome
    return Chromosome(resource_idx=chromosome.resource_idx, start_time=tuple(starts))


def repair(problem: Problem, chromosome: Chromosome) -> Chromosome:
    """Apply all feasible hard-constraint repairs in a stable order.

    Capacity constraints are intentionally not repaired here; they remain
    handled by the penalty model because fixing them may require changing
    resource assignments, which is left to selection and mutation.
    """
    repaired = repair_release_times(problem, chromosome)
    repaired = repair_precedence(problem, repaired)
    return repaired
