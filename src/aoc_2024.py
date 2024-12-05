import re
import collections
from typing import Annotated, DefaultDict
import typer
from pathlib import Path

cli = typer.Typer(
	no_args_is_help=True,
)


def get_path(*, day: int):
	return Path("inputs") / f"day{day}.input"


def day1_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=1),
):
	lines = [
		re.findall(pattern="\d+", string=line)
		for line in input_path.read_text().splitlines()
	]
	columns = ([], [])
	for first, second in lines:
		columns[0].append(int(first))
		columns[1].append(int(second))

	columns[0].sort()
	columns[1].sort()

	sum = 0
	for i in range(len(columns[0])):
		sum += abs(columns[0][i] - columns[1][i])
	print(sum)


def day1_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=1),
):
	lines = [
		re.findall(pattern="\d+", string=line)
		for line in input_path.read_text().splitlines()
	]
	columns = ([], [])
	for first, second in lines:
		columns[0].append(int(first))
		columns[1].append(int(second))

	frequency = collections.Counter(columns[1])

	sum = 0
	for i in range(len(columns[0])):
		sum += columns[0][i] * frequency[columns[0][i]]
	print(sum)


def day2_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=2),
):
	lines = input_path.read_text().splitlines()
	lines = [list(map(int, re.findall(pattern="\d+", string=line))) for line in lines]

	safe = 0
	for line in lines:
		direction = None
		is_safe = True
		for index in range(1, len(line)):
			diff = line[index] - line[index - 1]
			pos_diff = abs(diff)

			if pos_diff < 1 or pos_diff > 3:
				is_safe = False
				break

			sign = diff // pos_diff
			if direction and sign != direction:
				is_safe = False
				break
			direction = sign
		safe += 1 if is_safe else 0

	print(safe)


def day2_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=2),
):
	lines = input_path.read_text().splitlines()
	lines = [list(map(int, re.findall(pattern="\d+", string=line))) for line in lines]

	safe = 0
	for line in lines:
		direction = None
		is_safe = True
		allow_ignore = True
		for index in range(1, len(line)):
			diff = line[index] - line[index - 1]
			pos_diff = abs(diff)

			if pos_diff < 1 or pos_diff > 3:
				if allow_ignore:
					allow_ignore = False
					continue
				else:
					is_safe = False
					break

			sign = diff // pos_diff
			if direction and sign != direction:
				if allow_ignore:
					allow_ignore = False
					continue
				else:
					is_safe = False
					break
			direction = sign
		safe += 1 if is_safe else 0
	print(safe)


def day3_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=3),
):
	lines = input_path.read_text().splitlines()
	line = "".join(lines)
	solution = sum(
		[
			int(param[0]) * int(param[1])
			for param in re.findall(pattern=r"mul\((\d+),(\d+)\)", string=line)
		]
	)
	print(solution)


def day3_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=3),
):
	lines = input_path.read_text().splitlines()
	line = "".join(lines)

	matches = list(
		re.findall(pattern=r"(mul\((\d+),(\d+)\))|(do\(\))|(don't\(\))", string=line)
	)

	count = True
	sum = 0
	for _, left_mult, right_mult, is_do, is_dont in matches:
		if is_do:
			count = True
		elif is_dont:
			count = False
		elif count:
			sum += int(left_mult) * int(right_mult)

	print(sum)


class SafeGrid:
	def __init__(self, *, lines: list[str]):
		self.grid = [list(line) for line in lines]

	def _is_safe(self, x, y):
		if x < 0 or y < 0:
			return False
		if x >= len(self.grid[0]) or y >= len(self.grid):
			return False
		return True

	def __getitem__(self, key):
		x, y = key
		if not self._is_safe(x, y):
			return "."
		return self.grid[y][x]

	def __setitem__(self, key, value):
		x, y = key
		if self._is_safe(x, y):
			self.grid[y][x] = value

	def __iter__(self):
		for y in range(len(self.grid)):
			for x in range(len(self.grid[0])):
				yield x, y

	def __str__(self):
		output = ""
		for line in self.grid:
			output += f"{''.join(line)}\n"
		return output


def day4_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=4),
):
	grid = SafeGrid(lines=input_path.read_text().splitlines())

	xmas_count = 0
	for x, y in grid:
		for dir_x, dir_y in [
			(1, 0),
			(0, 1),
			(-1, 0),
			(0, -1),
			(1, 1),
			(-1, 1),
			(1, -1),
			(-1, -1),
		]:
			found_xmas = True
			for index, letter in enumerate("XMAS"):
				if grid[x + dir_x * index, y + dir_y * index] != letter:
					found_xmas = False
					break
			if found_xmas:
				xmas_count += 1
	print(xmas_count)


def day4_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=4),
):
	grid = SafeGrid(lines=input_path.read_text().splitlines())

	xmas_count = 0
	for x, y in grid:
		if grid[x, y] != "A":
			continue

		northeast = {"S", "M"}
		for dir_x, dir_y in [(1, 1), (-1, -1)]:
			northeast.discard(grid[x + dir_x, y + dir_y])

		if len(northeast) > 0:
			continue

		southeast = {"S", "M"}
		for dir_x, dir_y in [(-1, 1), (1, -1)]:
			southeast.discard(grid[x + dir_x, y + dir_y])

		if len(southeast) > 0:
			continue

		xmas_count += 1
	print(xmas_count)


def day5_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=5),
):
	lines = input_path.read_text().splitlines()
	index = list.index(lines, "")
	page_ordering_rules, updates = lines[:index], lines[index + 1 :]

	rules_before = collections.defaultdict(set)
	rules_after = collections.defaultdict(set)

	for rule in page_ordering_rules:
		number_must_be_before, number_must_be_after = map(int, rule.split("|"))
		rules_before[number_must_be_after].add(number_must_be_before)
		rules_after[number_must_be_before].add(number_must_be_after)

	updates = [list(map(int, update.split(","))) for update in updates]

	def check(update):
		"""O(n^2)"""
		for i in range(len(update)):
			elem = update[i]
			for j in range(len(update)):
				compare = update[j]
				if j < i:
					if compare in rules_after[elem]:
						return False
				elif j > i:
					if compare in rules_before[elem]:
						return False
		return True

	midpoint_sum = 0
	for update in updates:
		if check(update):
			midpoint_sum += update[len(update) // 2]
	print(midpoint_sum)

def day5_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=5),
):
	lines = input_path.read_text().splitlines()
	index = list.index(lines, "")
	page_ordering_rules, updates = lines[:index], lines[index + 1 :]

	rules_before = collections.defaultdict(set)
	rules_after = collections.defaultdict(set)

	for rule in page_ordering_rules:
		number_must_be_before, number_must_be_after = map(int, rule.split("|"))
		rules_before[number_must_be_after].add(number_must_be_before)
		rules_after[number_must_be_before].add(number_must_be_after)

	updates = [list(map(int, update.split(","))) for update in updates]

	def check(update: list[int]):
		"""O(n^2)"""
		for i in range(len(update)):
			elem = update[i]
			for j in range(len(update)):
				compare = update[j]
				if j < i:
					if compare in rules_after[elem]:
						return (i, j)
				elif j > i:
					if compare in rules_before[elem]:
						return (i, j)
		return None

	midpoint_sum = 0
	for update in updates:
		problem = check(update)
		should_sum = problem is not None
		while problem:
			x, y = problem
			update[x], update[y] = update[y], update[x]
			problem = check(update)
		if should_sum:
			midpoint_sum += update[len(update) // 2]
	print(midpoint_sum)

def day6_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=6),
):
	lines = input_path.read_text().splitlines()


def day6_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=6),
):
	lines = input_path.read_text().splitlines()


def day7_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=7),
):
	lines = input_path.read_text().splitlines()


def day7_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=7),
):
	lines = input_path.read_text().splitlines()


def day8_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=8),
):
	lines = input_path.read_text().splitlines()


def day8_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=8),
):
	lines = input_path.read_text().splitlines()


def day9_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=9),
):
	lines = input_path.read_text().splitlines()


def day9_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=9),
):
	lines = input_path.read_text().splitlines()


def day10_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=10),
):
	lines = input_path.read_text().splitlines()


def day10_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=10),
):
	lines = input_path.read_text().splitlines()


def day11_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=11),
):
	lines = input_path.read_text().splitlines()


def day11_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=11),
):
	lines = input_path.read_text().splitlines()


def day12_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=12),
):
	lines = input_path.read_text().splitlines()


def day12_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=12),
):
	lines = input_path.read_text().splitlines()


def day13_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=13),
):
	lines = input_path.read_text().splitlines()


def day13_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=13),
):
	lines = input_path.read_text().splitlines()


def day14_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=14),
):
	lines = input_path.read_text().splitlines()


def day14_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=14),
):
	lines = input_path.read_text().splitlines()


def day15_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=15),
):
	lines = input_path.read_text().splitlines()


def day15_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=15),
):
	lines = input_path.read_text().splitlines()


def day16_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=16),
):
	lines = input_path.read_text().splitlines()


def day16_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=16),
):
	lines = input_path.read_text().splitlines()


def day17_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=17),
):
	lines = input_path.read_text().splitlines()


def day17_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=17),
):
	lines = input_path.read_text().splitlines()


def day18_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=18),
):
	lines = input_path.read_text().splitlines()


def day18_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=18),
):
	lines = input_path.read_text().splitlines()


def day19_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=19),
):
	lines = input_path.read_text().splitlines()


def day19_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=19),
):
	lines = input_path.read_text().splitlines()


def day20_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=20),
):
	lines = input_path.read_text().splitlines()


def day20_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=20),
):
	lines = input_path.read_text().splitlines()


def day21_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=21),
):
	lines = input_path.read_text().splitlines()


def day21_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=21),
):
	lines = input_path.read_text().splitlines()


def day22_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=22),
):
	lines = input_path.read_text().splitlines()


def day22_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=22),
):
	lines = input_path.read_text().splitlines()


def day23_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=23),
):
	lines = input_path.read_text().splitlines()


def day23_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=23),
):
	lines = input_path.read_text().splitlines()


def day24_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=24),
):
	lines = input_path.read_text().splitlines()


def day24_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=24),
):
	lines = input_path.read_text().splitlines()


def day25_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=25),
):
	lines = input_path.read_text().splitlines()


def day25_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=25),
):
	lines = input_path.read_text().splitlines()


if __name__ == "__main__":
	for day in range(1, 26):
		cli.command()(globals().get(f"day{day}_part1"))
		cli.command()(globals().get(f"day{day}_part2"))
	cli()
