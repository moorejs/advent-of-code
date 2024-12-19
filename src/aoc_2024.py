import re
import collections
from typing import Annotated, Callable, DefaultDict, Optional
from dataclasses import dataclass
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
		re.findall(pattern=r"\d+", string=line)
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
		re.findall(pattern=r"\d+", string=line)
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
	lines = [list(map(int, re.findall(pattern=r"\d+", string=line))) for line in lines]

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
	lines = [list(map(int, re.findall(pattern=r"\d+", string=line))) for line in lines]

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

	def _is_safe(self, x: int, y: int):
		if x < 0 or y < 0:
			return False
		if x >= len(self.grid[0]) or y >= len(self.grid):
			return False
		return True

	def __getitem__(self, key: tuple[int, int]):
		x, y = key
		if not self._is_safe(x, y):
			return None
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


def walk(*, grid: SafeGrid, on_step: Optional[Callable] = None):
	dirs = [(0, 1), (-1, 0), (0, -1), (1, 0)]
	dir = 2  # cheating - first pos is ^

	x, y = None, None
	for grid_x, grid_y in grid:
		if grid[grid_x, grid_y] == "^":  # cheating - first pos is ^
			x, y = grid_x, grid_y
			break

	assert x is not None and y is not None

	unique = {(x, y)}
	unique_with_dir = {(x, y, dir)}
	while True:
		x += dirs[dir][0]
		y += dirs[dir][1]
		if not grid[x, y]:
			break
		if grid[x, y] == "#":
			x -= dirs[dir][0]
			y -= dirs[dir][1]
			dir = (dir + 1) % 4  # turn right
			continue
		unique.add((x, y))
		if on_step:
			on_step(x, y, dir)
		if (x, y, dir) in unique_with_dir:
			return None  # cycle
		unique_with_dir.add((x, y, dir))
	return unique


def day6_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=6),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	unique = walk(grid=grid)
	print(len(unique))


def day6_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=6),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	add_obstacle = set()

	def on_step(x, y, dir):
		save = grid[x, y]
		if save != ".":
			return
		grid[x, y] = "#"
		if not walk(grid=grid, on_step=None):
			print("cycle found", x, y)
			add_obstacle.add((x, y))
		grid[x, y] = save

	walk(grid=grid, on_step=on_step)
	print(len(add_obstacle))

def day7_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=7),
):
	lines = input_path.read_text().splitlines()
	lines = [
		re.search(pattern=r"(?P<sum>\d+): (?P<rest>.+)", string=line).groupdict()
		for line in lines
	]

	for line in lines:
		line["rest"] = list(map(int, line["rest"].split(" ")))

	def solve(numbers, index, acc, sum):
		if index == len(numbers):
			# print(acc, sum)
			return acc == sum
		addition = acc + numbers[index]
		multiplication = acc * numbers[index]
		return solve(numbers, index + 1, addition, sum) or solve(
			numbers, index + 1, multiplication, sum
		)

	final_sum = 0
	for line in lines:
		if solve(line["rest"], 1, line["rest"][0], int(line["sum"])):
			final_sum += int(line["sum"])
	print(final_sum)


def day7_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=7),
):
	lines = input_path.read_text().splitlines()
	lines = [
		re.search(pattern=r"(?P<sum>\d+): (?P<rest>.+)", string=line).groupdict()
		for line in lines
	]

	for line in lines:
		line["rest"] = list(map(int, line["rest"].split(" ")))

	def solve(numbers, index, acc, sum):
		if index == len(numbers):
			return acc == sum
		addition = acc + numbers[index]
		multiplication = acc * numbers[index]
		concat = int(str(acc) + str(numbers[index]))
		return (
			solve(numbers, index + 1, addition, sum)
			or solve(numbers, index + 1, multiplication, sum)
			or solve(numbers, index + 1, concat, sum)
		)

	final_sum = 0
	for line in lines:
		if solve(line["rest"], 1, line["rest"][0], int(line["sum"])):
			final_sum += int(line["sum"])
	print(final_sum)

def day8_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=8),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	nodes: DefaultDict[str, list[tuple[int, int]]] = collections.defaultdict(list)

	for x, y in grid:
		if grid[x, y] != ".":
			nodes[grid[x, y]].append((x, y))
	print(nodes)

	antinodes: set[tuple[int, int]] = set()
	for frequency, coordinates in nodes.items():
		# pairwise compare every coordinate
		for i in range(len(coordinates)):
			for j in range(len(coordinates)):
				if i == j:
					continue
				dx = coordinates[i][0] - coordinates[j][0]
				dy = coordinates[i][1] - coordinates[j][1]
				new_coord = (coordinates[i][0] + dx, coordinates[i][1] + dy)
				print(
					"comparing",
					i,
					j,
					coordinates[i],
					coordinates[j],
					dx,
					dy,
					grid[dx * 2, dy * 2],
				)
				if grid[new_coord] and grid[new_coord] != frequency:
					print(grid[new_coord], frequency)
					antinodes.add(new_coord)
	print(antinodes)
	print(len(antinodes))


def day8_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=8),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	nodes: DefaultDict[str, list[tuple[int, int]]] = collections.defaultdict(list)

	for x, y in grid:
		if grid[x, y] != ".":
			nodes[grid[x, y]].append((x, y))
	print(nodes)

	antinodes: set[tuple[int, int]] = set()
	for frequency, coordinates in nodes.items():
		# pairwise compare every coordinate
		for i in range(len(coordinates)):
			for j in range(len(coordinates)):
				if i == j:
					continue
				dx = coordinates[i][0] - coordinates[j][0]
				dy = coordinates[i][1] - coordinates[j][1]
				antinodes.add(coordinates[i])
				mult = 1
				while True:
					new_coord = (
						coordinates[i][0] + dx * mult,
						coordinates[i][1] + dy * mult,
					)
					if not grid[new_coord]:
						break
					if grid[new_coord] != frequency:
						antinodes.add(new_coord)
					mult += 1
				mult = -1
				while True:
					new_coord = (
						coordinates[i][0] + dx * mult,
						coordinates[i][1] + dy * mult,
					)
					if not grid[new_coord]:
						break
					if grid[new_coord] != frequency:
						antinodes.add(new_coord)
					mult -= 1

	print(len(antinodes))


@dataclass
class Section:
	id: int
	length: int


def day9_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=9),
):
	lines = input_path.read_text().splitlines()
	(line,) = lines
	# <file-length><free-space><file-length><free-space>...

	id = 0
	total = 0
	sections = []
	for index in range(len(line)):
		length = int(line[index])
		if index % 2 == 0:  # file length
			sections.append(Section(id=id, length=length))
			total += length
		else:
			id += 1
			sections.append(Section(id=-1, length=length))

	check_sum = 0
	section_index = 0
	section_reverse_index = 0
	for index in range(total):
		current_section = sections[section_index]
		while current_section.length <= 0:
			section_index += 1
			current_section = sections[section_index]

		if current_section.id == -1:  # if empty
			section_to_backfill_with = sections[-1 - section_reverse_index]
			section_to_backfill_with.length -= 1
			if section_to_backfill_with.length == 0:
				section_reverse_index += 2
			check_sum += index * section_to_backfill_with.id
		else:
			check_sum += index * current_section.id
		current_section.length -= 1
	print(check_sum)

def day9_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=9),
):
	lines = input_path.read_text().splitlines()
	(line,) = lines
	# <file-length><free-space><file-length><free-space>...

	id = 0
	total = 0
	sections = []
	for index in range(len(line)):
		length = int(line[index])
		if index % 2 == 0:  # file length
			sections.append(Section(id=id, length=length))
			total += length
		else:
			id += 1
			sections.append(Section(id=-1, length=length))

	new_list = sections[:]
	for i in range(len(sections) - 1, 0, -1):
		to_place = sections[i]
		if to_place.id == -1:
			continue

		for j in range(0, i):
			if sections[j].id == -1 and sections[j].length >= to_place.length:
				sections[j].length -= to_place.length
				print("placing", to_place.id, "at", j, "length", to_place.length)
				break

	# check_sum = 0
	# section_index = 0
	# section_reverse_index = 0
	# for index in range(total):
	# 	current_section = sections[section_index]
	# 	while current_section.length <= 0:
	# 		section_index += 1
	# 		current_section = sections[section_index]

	# 	if current_section.id == -1:  # if empty
	# 		section_to_backfill_with = sections[-1 - section_reverse_index]

	# 		if section_to_backfill_with.length <= current_section.length:
	# 			current_section.length -= section_to_backfill_with.length
	# 			section_reverse_index += 2
	# 		check_sum += index * section_to_backfill_with.id
	# 	else:
	# 		check_sum += index * current_section.id
	# 	current_section.length -= 1
	# print(check_sum)

def day10_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=10),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	starts = []
	for x, y in grid:
		if grid[x, y] == "0":
			starts.append((x, y))
	print(starts)

	score = 0
	for start_x, start_y in starts:
		nines = set()
		visited = set()
		neighbors = [(start_x, start_y, "0")]
		while len(neighbors):
			neighbor = neighbors.pop()
			x, y, value = neighbor
			if (x, y) in visited:
				continue
			visited.add((x, y))
			for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
				if grid[x + dir[0], y + dir[1]] == str(int(value) + 1):
					print(
						start_x, start_y, x, y, "reached", grid[x + dir[0], y + dir[1]]
					)
					if grid[x + dir[0], y + dir[1]] == "9":
						nines.add((x + dir[0], y + dir[1]))
					else:
						neighbors.append(
							(x + dir[0], y + dir[1], grid[x + dir[0], y + dir[1]])
						)
		score += len(nines)
	print(score)


def day10_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=10),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	starts = []
	for x, y in grid:
		if grid[x, y] == "0":
			starts.append((x, y))
	print(starts)

	score = 0
	for start_x, start_y in starts:
		uniques = collections.defaultdict(int)
		visited = set()
		count = 0
		neighbors = [(start_x, start_y, "0")]
		while len(neighbors):
			neighbor = neighbors.pop()
			x, y, value = neighbor
			# if (x, y) in visited:
			# continue
			# visited.add((x, y))
			for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
				if grid[x + dir[0], y + dir[1]] == str(int(value) + 1):
					print(
						start_x, start_y, x, y, "reached", grid[x + dir[0], y + dir[1]]
					)
					if grid[x + dir[0], y + dir[1]] == "9":
						uniques[x + dir[0], y + dir[1]] += 1
						count += 1
					else:
						neighbors.append(
							(x + dir[0], y + dir[1], grid[x + dir[0], y + dir[1]])
						)
		score += count
		print(uniques)
	print(score)


def day11_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=11),
):
	lines = input_path.read_text().splitlines()
	line = lines[0]
	stones = list(map(int, re.findall(pattern=r"\d+", string=line)))

	def apply_rules(stone: int):
		if stone == 0:
			return [1]
		as_str = str(stone)
		if len(as_str) % 2 == 0:
			half = len(as_str) // 2
			return [int(as_str[:half]), int(as_str[half:])]
		return [stone * 2024]

	for _blink in range(25):
		new_stones = []
		for stone in stones:
			new_stones.extend(apply_rules(stone))
		stones = new_stones
	print(len(stones))


def day11_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=11),
):
	lines = input_path.read_text().splitlines()
	line = lines[0]
	stones = list(map(int, re.findall(pattern=r"\d+", string=line)))

	cache = collections.defaultdict(int)

	def apply_rules(stone: int, left: int):
		if cache[stone, left]:
			return cache[stone, left]
		if left == 0:
			return 1
		if stone == 0:
			return 0 + apply_rules(1, left - 1)
		as_str = str(stone)
		if len(as_str) % 2 == 0:
			half = len(as_str) // 2

			right_half_arg = int(as_str[:half])
			right_half = apply_rules(right_half_arg, left - 1)
			left_half_arg = int(as_str[half:])
			left_half = apply_rules(left_half_arg, left - 1)

			cache[right_half_arg, left - 1] = right_half
			cache[left_half_arg, left - 1] = left_half

			return 0 + right_half + left_half
		return 0 + apply_rules(stone * 2024, left - 1)

	print(sum(map(lambda stone: apply_rules(stone, 75), stones)))
	# print(sum(map(lambda stone: apply_rules(stone, 25), [125, 17])))


def day12_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=12),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	for x, y in grid:
		pass

	# perimeter increases for each non-equal neighbor
	# area increases for each equal cell

	# price = area * perimeter

	total_price = 0
	visited = set()
	for x, y in grid:
		if (x, y) in visited:
			continue
		neighbors = [(x, y)]
		perimeter = 0
		area = 0
		while len(neighbors):
			node_x, node_y = neighbors.pop()
			if (node_x, node_y) in visited:
				continue
			visited.add((node_x, node_y))
			area += 1
			for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
				next_coord = (node_x + dir[0], node_y + dir[1])
				if grid[next_coord] != grid[x, y]:
					perimeter += 1
				else:
					neighbors.append(next_coord)
		print(x, y, area, perimeter)
		price = area * perimeter
		total_price += price
	print(total_price)


def day12_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=12),
):
	lines = input_path.read_text().splitlines()

import functools

def day13_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=13),
):
	text = input_path.read_text()

	@functools.lru_cache(maxsize=None)
	def solve(x_left: int, y_left: int, delta_a, delta_b):
		if x_left < 0 or y_left < 0:
			return float("inf")
		if x_left == 0 and y_left == 0:
			return 0
		return min(
			3 + solve(x_left - delta_a[0], y_left - delta_a[1], delta_a, delta_b),
			1 + solve(x_left - delta_b[0], y_left - delta_b[1], delta_a, delta_b),
		)

	# x(a,b), where a is the number of times the machine A is used and b is the number of times the machine B is used
	# x(a,b) = min(3 + x(a-1,b), 1 + x(a,b-1))

	machines = text.split("\n\n")

	sum_of_min_for_all_machines = 0
	for machine in machines:
		delta_a = tuple(map(int, re.search(r"A: X\+(\d+), Y\+(\d+)", machine).groups()))
		delta_b = tuple(map(int, re.search(r"B: X\+(\d+), Y\+(\d+)", machine).groups()))
		prize = re.search(r"Prize: X=(\d+), Y=(\d+)", machine).groups()

		machine_min = solve(int(prize[0]), int(prize[1]), delta_a, delta_b)
		if machine_min != float("inf"):
			sum_of_min_for_all_machines += machine_min
	print(sum_of_min_for_all_machines)
	return sum_of_min_for_all_machines


def day13_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=13),
):
	text = input_path.read_text()

	@functools.lru_cache(maxsize=None)
	def solve(x_left: int, y_left: int, delta_a, delta_b):
		if x_left < 0 or y_left < 0:
			return float("inf")
		if x_left == 0 and y_left == 0:
			return 0
		return min(
			3 + solve(x_left - delta_a[0], y_left - delta_a[1], delta_a, delta_b),
			1 + solve(x_left - delta_b[0], y_left - delta_b[1], delta_a, delta_b),
		)

	# x(a,b), where a is the number of times the machine A is used and b is the number of times the machine B is used
	# x(a,b) = min(3 + x(a-1,b), 1 + x(a,b-1))

	machines = text.split("\n\n")

	sum_of_min_for_all_machines = 0
	for machine in machines:
		delta_a = tuple(map(int, re.search(r"A: X\+(\d+), Y\+(\d+)", machine).groups()))
		delta_b = tuple(map(int, re.search(r"B: X\+(\d+), Y\+(\d+)", machine).groups()))
		prize = re.search(r"Prize: X=(\d+), Y=(\d+)", machine).groups()

		machine_min = solve(
			int(prize[0]) + 10_000_000_000_000,
			int(prize[1]) + 10_000_000_000_000,
			delta_a,
			delta_b,
		)
		if machine_min != float("inf"):
			sum_of_min_for_all_machines += machine_min
	print(sum_of_min_for_all_machines)
	return sum_of_min_for_all_machines


@dataclass
class Vector2:
	x: int
	y: int


@dataclass
class Robot:
	position: Vector2
	velocity: Vector2

def day14_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=14),
):
	lines = input_path.read_text().splitlines()

	robots: list[Robot] = []
	for line in lines:
		match = re.match(
			pattern=r"p=(?P<p_x>-?\d+),(?P<p_y>-?\d+) v=(?P<v_x>-?\d+),(?P<v_y>-?\d+)",
			string=line,
		)
		p_x, p_y, v_x, v_y = map(int, list(match.groups()))
		robots.append(Robot(position=Vector2(p_x, p_y), velocity=Vector2(v_x, v_y)))

	size = Vector2(101, 103)
	for _ in range(100):
		for robot in robots:
			robot.position.x = (robot.position.x + robot.velocity.x) % size.x
			robot.position.y = (robot.position.y + robot.velocity.y) % size.y

	position_to_count = DefaultDict(int)

	for robot in robots:
		print(robot.position)
		if robot.position.x == size.x // 2 or robot.position.y == size.y // 2:
			continue
		x_quad = (2 * robot.position.x) // size.x
		y_quad = (2 * robot.position.y) // size.y
		position_to_count[x_quad, y_quad] += 1

	product = 1
	for value in position_to_count.values():
		product *= value
	print(product)


def day14_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=14),
):
	lines = input_path.read_text().splitlines()

	robots: list[Robot] = []
	for line in lines:
		match = re.match(
			pattern=r"p=(?P<p_x>-?\d+),(?P<p_y>-?\d+) v=(?P<v_x>-?\d+),(?P<v_y>-?\d+)",
			string=line,
		)
		p_x, p_y, v_x, v_y = map(int, list(match.groups()))
		robots.append(Robot(position=Vector2(p_x, p_y), velocity=Vector2(v_x, v_y)))

	size = Vector2(101, 103)

	seconds = 0
	while True:
		position_to_count = DefaultDict(int)
		for robot in robots:
			if robot.position.x == size.x // 2 or robot.position.y == size.y // 2:
				continue
			x_quad = (2 * robot.position.x) // size.x
			y_quad = (2 * robot.position.y) // size.y
			position_to_count[x_quad, y_quad] += 1

		if True:
			print(seconds)
			pos = {(robot.position.x, robot.position.y) for robot in robots}
			for y in range(size.y):
				for x in range(size.x):
					if (x, y) in pos:
						print("#", end="")
					else:
						print(".", end="")
				print()

		if seconds > size.x * size.y:
			break

		for robot in robots:
			robot.position.x = (robot.position.x + robot.velocity.x) % size.x
			robot.position.y = (robot.position.y + robot.velocity.y) % size.y
		seconds += 1

		# output stdout to file and grep -B 100 "######" to find answer

def day15_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=15),
):
	map, moves = input_path.read_text().split("\n\n")

	grid = SafeGrid(lines=map.splitlines())
	moves = " ".join(moves.splitlines())

	pos = None
	for x, y in grid:
		if grid[x, y] == "@":
			pos = Vector2(x, y)
			break

	print(grid)

	for move in moves:
		print("move", move)

		if move == "<":
			# when pushing left, look for first "."
			# if "#" is found before "." or "O", you can't move
			# if "O" is found before ".", you can move but have to move the O
			can_move = True
			look_ahead_coord = Vector2(pos.x - 1, pos.y)
			while True:
				look_ahead = grid[look_ahead_coord.x, look_ahead_coord.y]
				if look_ahead == ".":
					break
				if look_ahead == "#":
					can_move = False
					break
				look_ahead_coord.x -= 1
			if can_move:
				for x in range(pos.x - 2, look_ahead_coord.x - 1, -1):
					# print("drawing", x, pos.y)
					grid[x, pos.y] = "O"
				grid[pos.x - 1, pos.y] = "@"
				grid[pos.x, pos.y] = "."
				pos.x -= 1
		if move == ">":
			# when pushing left, look for first "."
			# if "#" is found before "." or "O", you can't move
			# if "O" is found before ".", you can move but have to move the O
			can_move = True
			look_ahead_coord = Vector2(pos.x + 1, pos.y)
			while True:
				look_ahead = grid[look_ahead_coord.x, look_ahead_coord.y]
				if look_ahead == ".":
					break
				if look_ahead == "#":
					can_move = False
					break
				look_ahead_coord.x += 1
			print(can_move, pos, look_ahead_coord)
			if can_move:
				for x in range(pos.x + 2, look_ahead_coord.x + 1, 1):
					grid[x, pos.y] = "O"
				grid[pos.x + 1, pos.y] = "@"
				grid[pos.x, pos.y] = "."
				pos.x += 1
		if move == "v":
			can_move = True
			look_ahead_coord = Vector2(pos.x, pos.y + 1)
			while True:
				look_ahead = grid[look_ahead_coord.x, look_ahead_coord.y]
				if look_ahead == ".":
					break
				if look_ahead == "#":
					can_move = False
					break
				look_ahead_coord.y += 1
			print(can_move, pos, look_ahead_coord)
			if can_move:
				for y in range(pos.y + 2, look_ahead_coord.y + 1, 1):
					grid[pos.x, y] = "O"
				grid[pos.x, pos.y + 1] = "@"
				grid[pos.x, pos.y] = "."
				pos.y += 1
		if move == "^":
			can_move = True
			look_ahead_coord = Vector2(pos.x, pos.y - 1)
			while True:
				look_ahead = grid[look_ahead_coord.x, look_ahead_coord.y]
				if look_ahead == ".":
					break
				if look_ahead == "#":
					can_move = False
					break
				look_ahead_coord.y -= 1
			print(can_move, pos, look_ahead_coord)
			if can_move:
				for y in range(pos.y - 2, look_ahead_coord.y - 1, -1):
					grid[pos.x, y] = "O"
				grid[pos.x, pos.y - 1] = "@"
				grid[pos.x, pos.y] = "."
				pos.y -= 1
		# print(grid)

	sum = 0
	for x, y in grid:
		if grid[x, y] == "O":
			print(x, y)
			sum += y * 100 + x
	print(sum)

	print(moves)


def day15_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=15),
):
	lines = input_path.read_text().splitlines()
	pass


def day16_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=16),
):
	lines = input_path.read_text().splitlines()
	grid = SafeGrid(lines=lines)

	start = None
	end = None
	for x, y in grid:
		if grid[x, y] == "S":
			start = (x, y)
		if grid[x, y] == "E":
			end = (x, y)

	dist = {}
	prev = {}
	count = {}
	for x, y in grid:
		dist[x, y] = float("inf")
		prev[x, y] = (start[0] - 1, start[1])
	dist[start] = 0
	count[start] = 1

	Q = set([(x, y) for x, y in grid])
	while len(Q):
		u = min(Q, key=lambda pos: dist[pos])
		Q.remove(u)
		if u == end:
			break
		for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
			prev_dir = (u[0] - prev[u][0], u[1] - prev[u][1])
			if prev_dir == dir:
				cost = 1
			elif prev_dir == (-dir[0], -dir[1]):
				cost = float("inf")
			else:
				cost = 1001
			alt = dist[u] + cost
			# print(u, dist[u], cost, prev_dir, dir)
			v = (u[0] + dir[0], u[1] + dir[1])
			if grid[v] == "#":
				continue
			if alt < dist[v]:
				dist[v] = alt
				prev[v] = u
				count[v] = 1
			elif alt == dist[v]:
				count[v] += 1

	follow = prev[end]
	sum = count[end]
	while follow != start:
		follow = prev[follow]
		sum += count[follow]

	print(count)

	# prev[u]
	# alt = dist[u] + cost(u, v)
	# if alt < dist[u]:
	# dist[u] = alt
	# prev[u] = u

	# print(start, end)

	# paths = []
	# options = [(start, 0, [start], (1, 0))]
	# while len(options):
	# 	# options.sort(key=lambda x: x[1])
	# 	# print(options)
	# 	pos, cost, path, dir = options.pop()
	# 	# print("pos", pos, "cost", cost, "path", path)
	# 	if pos == end:
	# 		# print("found", cost, path)
	# 		print(cost)
	# 		paths.append((cost, path))
	# 		break
	# 		continue
	# 		# break
	# 	for option_dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
	# 		new_pos = (pos[0] + option_dir[0], pos[1] + option_dir[1])

	# 		if grid[new_pos] == "#":
	# 			continue
	# 		# print("new_pos", new_pos)

	# 		if dir == option_dir and new_pos not in path:
	# 			# print("straight")
	# 			options.append((new_pos, cost + 1, path + [new_pos], option_dir))
	# 		# TODO: turn around == 2000
	# 		elif new_pos not in path:
	# 			# print("turn")
	# 			options.append((new_pos, cost + 1001, path + [new_pos], option_dir))

	# for cost, path in paths:
	# 	print()
	# 	print()
	# 	print(cost)
	# 	last_y = None
	# 	for x, y in grid:
	# 		if last_y != y:
	# 			print()
	# 		last_y = y
	# 		if grid[x, y] == "#":
	# 			print("#", end="")
	# 		elif (x, y) in path:
	# 			print("x", end="")
	# 		else:
	# 			print(".", end="")

def day16_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=16),
):
	lines = input_path.read_text().splitlines()

def run(program, a: int, b: int, c: int, goal=False):
	_a = a
	_b = b
	_c = c

	def get_combo_value(op: int):
		if op <= 3:
			return op
		if op == 4:
			return _a
		if op == 5:
			return _b
		if op == 6:
			return _c
		assert False

	ip = 0
	output = []
	while ip < len(program):
		opcode = program[ip]
		operand = program[ip + 1]
		# print(f"{ip}: op={opcode} in={operand}, a={_a}, b={_b}, c={_c}")
		match opcode:
			case 0:
				_a = _a // (2 ** get_combo_value(operand))
			case 1:
				_b = _b ^ operand
			case 2:
				_b = get_combo_value(operand) % 8
			case 3:
				if _a != 0:
					ip = operand
					continue  # do not increment ip
			case 4:
				_b = _b ^ _c
			case 5:
				new_output = get_combo_value(operand) % 8
				if goal and new_output != program[len(output)]:
					return output
				output.append(new_output)
				# if new_output == 4:
				# 	print(a, b, c)
				# if len(output) == 2 and output[0] == 2 and output[1] == 4:
				# 	print(a, b, c)
				# return []
			case 6:
				_b = _a // (2 ** get_combo_value(operand))
			case 7:
				_c = _a // (2 ** get_combo_value(operand))

		ip += 2
	return output


def day17_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=17),
):
	lines = input_path.read_text().splitlines()

	a: int = int(re.search(pattern=r"\d+", string=lines[0]).group())
	b: int = int(re.search(pattern=r"\d+", string=lines[1]).group())
	c: int = int(re.search(pattern=r"\d+", string=lines[2]).group())
	program = list(map(int, re.findall(pattern=r"\d+", string=lines[4])))

	output = run(program, a, b, c)
	print(",".join(map(str, output)))

def day17_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=17),
):
	lines = input_path.read_text().splitlines()
	program = list(map(int, re.findall(pattern=r"\d+", string=lines[4])))

	a = 35_000_000_000_000
	upper_bound = 282_000_000_000_000
	while a < upper_bound:
		output = run(program, a, 0, 0, goal=False)
		# if len(output):
		if len(output) == len(program):
			print(a, output, len(output), len(program))

			match = True
			for i in range(len(output)):
				if output[i] != program[i]:
					match = False
					break
			if match:
				print(a)
				break
		a += 100_000_000
		# if a % 100_000_000_000 == 0:
		# print(a)
		if a > upper_bound:
			break


def day18_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=18),
):
	lines = input_path.read_text().splitlines()

	size = 71
	ticks = 1024

	grid = SafeGrid(lines=["." * size for _ in range(size)])

	for _ in range(ticks):
		line = lines[_]
		x, y = map(int, re.search(r"(\d+),(\d+)", line).groups())
		grid[x, y] = "#"

	print(grid)

	distances = {(x, y): float("inf") for x in range(size) for y in range(size)}
	distances[0, 0] = 0

	unvisited = set(distances.keys())
	while len(unvisited):
		# find the unvisited node with the smallest distance
		current = min(unvisited, key=lambda pos: distances[pos])
		print(current)
		if distances[current] == float("inf"):
			break

		# mark distances to any visitable neighbors
		for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
			neighbor = (current[0] + dir[0], current[1] + dir[1])
			if grid[neighbor] == ".":
				new_distance = distances[current] + 1
				if new_distance < distances[neighbor]:
					distances[neighbor] = new_distance

		# mark visited
		unvisited.remove(current)

	print(distances[size - 1, size - 1])


def search(grid, size=71):
	distances = {(x, y): float("inf") for x in range(size) for y in range(size)}
	distances[0, 0] = 0

	unvisited = set(distances.keys())
	while len(unvisited):
		# find the unvisited node with the smallest distance
		current = min(unvisited, key=lambda pos: distances[pos])
		if distances[current] == float("inf"):
			return distances

		# mark distances to any visitable neighbors
		for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
			neighbor = (current[0] + dir[0], current[1] + dir[1])
			if grid[neighbor] == ".":
				new_distance = distances[current] + 1
				if new_distance < distances[neighbor]:
					distances[neighbor] = new_distance

		# mark visited
		unvisited.remove(current)

	return distances


def day18_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=18),
):
	lines = input_path.read_text().splitlines()

	size = 71

	grid = SafeGrid(lines=["." * size for _ in range(size)])

	ticks = 0
	while True:
		line = lines[ticks]
		ticks += 1
		x, y = map(int, re.search(r"(\d+),(\d+)", line).groups())
		grid[x, y] = "#"
		if ticks < 2800:
			continue
		if search(grid)[size - 1, size - 1] == float("inf"):
			break
		print(ticks)
	print(x, y)


def day19_part1(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=19),
):
	pattern, lines = input_path.read_text().split("\n\n")
	lines = lines.splitlines()

	towels = pattern.split(", ")
	# towels.sort(key=lambda string: len(string), reverse=True)

	base = {}
	for towel in towels:
		trie = base
		for letter in towel:
			if letter not in trie:
				trie[letter] = {}
			trie = trie[letter]
		trie["."] = None
	print(base)

	for key in base:
		print(key, base[key])

	# def is_possible(line):
	# 	if len(line) == 0:
	# 		return True
	# 	trie = base
	# 	index = 0
	# 	if line[index] not in trie:
	# 		return False
	# 	while line[index] in trie:
	# 		trie = trie[line[index]]
	# 		if trie[line[index]]["."]:
	# 			return is_possible(line[index + 1 :])
	# 		index += 1

	def is_possible(line):
		# find all possible paths from first letter
		# if line[0] not in base:
		# 	return False
		if len(line) == 0:
			return 1

		trie = base
		index = 0
		while index < len(line):
			trie = trie.get(line[index], {})
			index += 1
			if "." in trie:
				try_ending_here = is_possible(line[index:])
				if try_ending_here:
					return 1
		return 0

	print(sum([is_possible(line) for line in lines]))
	# for line in lines:
	# print(is_possible(line))


def day19_part2(
	input_path: Annotated[
		Path, typer.Option(exists=True, dir_okay=False, readable=True)
	] = get_path(day=19),
):
	lines = input_path.read_text().splitlines()
	pattern, lines = input_path.read_text().split("\n\n")
	lines = lines.splitlines()

	towels = pattern.split(", ")
	# towels.sort(key=lambda string: len(string), reverse=True)

	base = {}
	for towel in towels:
		trie = base
		for letter in towel:
			if letter not in trie:
				trie[letter] = {}
			trie = trie[letter]
		trie["."] = None
	print(base)

	for key in base:
		print(key, base[key])

	# @functools.lru_cache(maxsize=None)
	def unique_trails_count(line):
		# find all possible paths from first letter
		# if line[0] not in base:
		# 	return False
		# if len(line) == 0:
		# 	return 0
		# print(line)

		trie = base
		index = 0
		total = 0
		while index < len(line):
			# print(trie)
			if line[index] not in trie:
				break
			trie = trie.get(line[index], {})
			index += 1
			if "." in trie:
				# print("can try:", line[index:])
				# pass
				if index != len(line):
					try_ending_here = unique_trails_count(line[index:])
					total += 1 + try_ending_here
		print(line, total)
		return total

	# 690 too low
	# print(sum([unique_trails_count(line) for line in lines]))

	print(unique_trails_count("gbbr"))
	# print(unique_trails_count("gbbr"))
	# for line in lines:
	# print(unique_trails_count(line))

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