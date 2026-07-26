"""Structure-aware parsing for ordinary LaTeX resumes.

The parser deliberately relies on LaTeX sections, list environments, and the
commands already present in the uploaded source.  It never requires template
annotations or rewrites the document around a new template.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


SECTION_ALIASES = {
    "summary": {"professional summary", "summary"},
    "experience": {"experience", "professional experience", "work experience"},
    "skills": {"skills", "technical skills", "core skills"},
    "projects": {"projects", "selected projects", "personal projects"},
}


class LatexStructureError(ValueError):
    """Raised when a structural edit target cannot be identified safely."""


@dataclass(frozen=True)
class TextRange:
    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True)
class LatexSection:
    category: str
    title: str
    command_range: TextRange
    content_range: TextRange


@dataclass(frozen=True)
class LatexItem:
    command: str
    content_range: TextRange
    block_range: TextRange
    depth: int
    ordinal: int

    def content(self, source: str) -> str:
        return self.content_range.text(source)


@dataclass(frozen=True)
class ProjectEntry:
    name: str
    block_range: TextRange
    field_ranges: tuple[TextRange, ...]
    bullets: tuple[LatexItem, ...]
    command: str


@dataclass(frozen=True)
class ResumeStructure:
    source: str
    sections: dict[str, LatexSection]
    summary_range: TextRange
    experience_bullets: tuple[LatexItem, ...]
    skill_items: tuple[LatexItem, ...]
    project_entries: tuple[ProjectEntry, ...]


def parse_resume_structure(
    source: str,
    *,
    require_projects: bool = False,
) -> ResumeStructure:
    """Locate assignment-approved targets in an uploaded LaTeX document."""

    sections = _locate_sections(source)
    required = ("summary", "experience", "skills")
    missing = [name for name in required if name not in sections]
    if require_projects and "projects" not in sections:
        missing.append("projects")
    if missing:
        raise LatexStructureError(
            "Could not safely identify required LaTeX section(s): "
            + ", ".join(missing)
            + ". Use a conventional section heading such as Summary, Experience, "
            "Skills, or Projects."
        )

    summary_range = _summary_content_range(source, sections["summary"])
    experience_bullets = tuple(
        _section_bullets(source, sections["experience"], category="experience")
    )
    if len(experience_bullets) < 2:
        raise LatexStructureError(
            "Could not safely identify at least two existing experience bullets "
            "from LaTeX list structure."
        )
    skill_items = tuple(_section_items(source, sections["skills"]))
    if not skill_items:
        raise LatexStructureError(
            "Could not safely identify an editable item in the Skills section."
        )
    project_entries: tuple[ProjectEntry, ...] = ()
    if "projects" in sections:
        project_entries = tuple(_project_entries(source, sections["projects"]))
        if require_projects and not project_entries:
            raise LatexStructureError(
                "Could not safely identify individual entries in the Projects section."
            )
    return ResumeStructure(
        source=source,
        sections=sections,
        summary_range=summary_range,
        experience_bullets=experience_bullets,
        skill_items=skill_items,
        project_entries=project_entries,
    )


def balanced_brace_content(source: str, opening: int) -> tuple[str, int]:
    """Return a braced argument and the index immediately after its closing brace."""

    if opening < 0 or opening >= len(source) or source[opening] != "{":
        raise LatexStructureError("Expected a LaTeX command argument.")
    depth = 0
    escaped = False
    for index in range(opening, len(source)):
        char = source[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index], index + 1
    raise LatexStructureError("Unbalanced braces in resume LaTeX.")


def latex_to_plain(text: str) -> str:
    """Convert a small LaTeX fragment to comparison-friendly text."""

    plain = re.sub(r"(?m)(?<!\\)%.*$", " ", text)
    plain = re.sub(r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1", plain)
    plain = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", plain)
    plain = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", plain)
    plain = (
        plain.replace(r"\&", "&")
        .replace(r"\%", "%")
        .replace(r"\_", "_")
        .replace("~", " ")
    )
    plain = re.sub(r"[{}$]", " ", plain)
    return re.sub(r"\s+", " ", plain).strip()


def _locate_sections(source: str) -> dict[str, LatexSection]:
    matches: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\\section\*?\s*\{", source):
        if _is_commented(source, match.start()):
            continue
        opening = source.find("{", match.start(), match.end())
        title, end = balanced_brace_content(source, opening)
        matches.append((match.start(), end, latex_to_plain(title).casefold()))
    located: dict[str, LatexSection] = {}
    for index, (start, command_end, normalized_title) in enumerate(matches):
        category = next(
            (
                name
                for name, aliases in SECTION_ALIASES.items()
                if normalized_title in aliases
            ),
            None,
        )
        if category is None:
            continue
        if category in located:
            raise LatexStructureError(
                f"Multiple plausible {category} sections were found; refusing an "
                "ambiguous edit."
            )
        end = matches[index + 1][0] if index + 1 < len(matches) else _document_end(
            source, command_end
        )
        located[category] = LatexSection(
            category=category,
            title=normalized_title,
            command_range=TextRange(start, command_end),
            content_range=TextRange(command_end, end),
        )
    return located


def _document_end(source: str, start: int) -> int:
    match = re.search(r"\\end\s*\{document\}", source[start:])
    return start + match.start() if match else len(source)


def _summary_content_range(source: str, section: LatexSection) -> TextRange:
    start, end = section.content_range.start, section.content_range.end
    fragment = source[start:end]
    items = _items_in_range(source, start, end)
    if items:
        if len(items) != 1:
            raise LatexStructureError(
                "The Summary section contains multiple list items and is ambiguous."
            )
        return items[0].content_range
    lines = list(re.finditer(r"(?m)^.*(?:\n|$)", fragment))
    meaningful = [
        line
        for line in lines
        if line.group(0).strip()
        and not line.group(0).lstrip().startswith("%")
    ]
    if not meaningful:
        raise LatexStructureError("The Summary section contains no editable text.")
    first = start + meaningful[0].start()
    last = start + meaningful[-1].end()
    while last > first and source[last - 1] in "\r\n":
        last -= 1
    candidate = source[first:last].strip()
    if re.search(r"\\(?:begin|end)\s*\{", candidate):
        raise LatexStructureError(
            "The Summary section structure is ambiguous; no safe prose target was found."
        )
    leading = len(source[first:last]) - len(source[first:last].lstrip())
    trailing = len(source[first:last]) - len(source[first:last].rstrip())
    return TextRange(first + leading, last - trailing)


def _section_bullets(
    source: str, section: LatexSection, *, category: str
) -> list[LatexItem]:
    items = _items_in_range(
        source, section.content_range.start, section.content_range.end
    )
    macro_items = [item for item in items if item.command.lower() != "item"]
    if macro_items:
        return _renumber(macro_items)
    if not items:
        return []
    deepest = max(item.depth for item in items)
    candidates = [
        item
        for item in items
        if item.depth == deepest
        and latex_to_plain(item.content(source))
        and not re.search(
            r"\\(?:begin\{tabular|resumeEntry)\b", item.content(source)
        )
    ]
    if category == "experience" and len(candidates) < 2:
        return []
    return _renumber(candidates)


def _section_items(source: str, section: LatexSection) -> list[LatexItem]:
    items = _items_in_range(
        source, section.content_range.start, section.content_range.end
    )
    if not items:
        return []
    macro_items = [item for item in items if item.command.lower() != "item"]
    return _renumber(macro_items or items)


def _items_in_range(source: str, start: int, end: int) -> list[LatexItem]:
    fragment = source[start:end]
    environment_events = _environment_depth_events(fragment)
    commands: list[tuple[int, str, TextRange, int]] = []
    pattern = re.compile(r"\\(?P<command>resumeItem|item)\b")
    for match in pattern.finditer(fragment):
        absolute = start + match.start()
        if _is_commented(source, absolute):
            continue
        command = match.group("command")
        cursor = start + match.end()
        while cursor < end and source[cursor].isspace():
            cursor += 1
        if cursor < end and source[cursor] == "[":
            closing = source.find("]", cursor + 1, end)
            if closing < 0:
                raise LatexStructureError("Unbalanced optional item argument.")
            cursor = closing + 1
            while cursor < end and source[cursor].isspace():
                cursor += 1
        if cursor < end and source[cursor] == "{":
            _, argument_end = balanced_brace_content(source, cursor)
            content_range = TextRange(cursor + 1, argument_end - 1)
        elif wrapper := re.match(r"\\(?:small|footnotesize)\s*\{", source[cursor:end]):
            opening = cursor + wrapper.end() - 1
            _, argument_end = balanced_brace_content(source, opening)
            content_range = TextRange(opening + 1, argument_end - 1)
        else:
            content_start = cursor
            content_range = TextRange(content_start, content_start)
        depth = _depth_at(environment_events, match.start())
        commands.append((absolute, command, content_range, depth))

    results: list[LatexItem] = []
    for index, (absolute, command, content_range, depth) in enumerate(commands):
        if content_range.start == content_range.end:
            candidates = [
                position
                for position, _, _, other_depth in commands[index + 1 :]
                if other_depth <= depth
            ]
            env_end = _matching_itemize_end(source, content_range.start, end, depth)
            block_end = min([*candidates, env_end, end])
            content_end = block_end
            while content_end > content_range.start and source[content_end - 1].isspace():
                content_end -= 1
            content_range = TextRange(content_range.start, content_end)
        next_peer = next(
            (
                position
                for position, _, _, other_depth in commands[index + 1 :]
                if other_depth <= depth
            ),
            end,
        )
        block_end = min(next_peer, end)
        results.append(
            LatexItem(
                command=command,
                content_range=content_range,
                block_range=TextRange(absolute, block_end),
                depth=depth,
                ordinal=len(results) + 1,
            )
        )
    return results


def _project_entries(source: str, section: LatexSection) -> list[ProjectEntry]:
    start, end = section.content_range.start, section.content_range.end
    resume_entries: list[tuple[int, tuple[TextRange, ...]]] = []
    for match in re.finditer(r"\\resumeEntry\b", source[start:end]):
        absolute = start + match.start()
        if _is_commented(source, absolute):
            continue
        cursor = start + match.end()
        fields: list[TextRange] = []
        for _ in range(4):
            while cursor < end and source[cursor].isspace():
                cursor += 1
            if cursor >= end or source[cursor] != "{":
                fields = []
                break
            _, argument_end = balanced_brace_content(source, cursor)
            fields.append(TextRange(cursor + 1, argument_end - 1))
            cursor = argument_end
        if fields:
            resume_entries.append((absolute, tuple(fields)))
    if resume_entries:
        entries: list[ProjectEntry] = []
        for index, (entry_start, fields) in enumerate(resume_entries):
            entry_end = (
                resume_entries[index + 1][0]
                if index + 1 < len(resume_entries)
                else end
            )
            block_start = _line_start(source, entry_start)
            block_end = entry_end
            while block_end > block_start and source[block_end - 1] in " \t\r\n":
                block_end -= 1
            bullets = tuple(
                item
                for item in _items_in_range(source, fields[-1].end, entry_end)
                if item.command.lower() != "item" or item.depth > 0
            )
            entries.append(
                ProjectEntry(
                    name=latex_to_plain(fields[0].text(source)),
                    block_range=TextRange(block_start, block_end),
                    field_ranges=fields,
                    bullets=bullets,
                    command="resumeEntry",
                )
            )
        return entries

    items = _items_in_range(source, start, end)
    if not items:
        return []
    shallowest = min(item.depth for item in items)
    roots = [item for item in items if item.depth == shallowest]
    entries = []
    for item in roots:
        body = item.content(source)
        title_match = re.search(r"\\textbf\s*\{", body)
        if not title_match:
            continue
        opening = item.content_range.start + title_match.end() - 1
        title, title_end = balanced_brace_content(source, opening)
        nested = tuple(
            candidate
            for candidate in items
            if item.block_range.start < candidate.block_range.start < item.block_range.end
            and candidate.depth > item.depth
        )
        entries.append(
            ProjectEntry(
                name=latex_to_plain(title),
                block_range=item.block_range,
                field_ranges=(TextRange(opening + 1, title_end - 1),),
                bullets=nested,
                command="item",
            )
        )
    return entries


def _environment_depth_events(fragment: str) -> list[tuple[int, int]]:
    depth = 0
    events: list[tuple[int, int]] = [(0, 0)]
    for match in re.finditer(r"\\(?P<kind>begin|end)\s*\{itemize\}", fragment):
        if match.group("kind") == "begin":
            depth += 1
        else:
            depth = max(0, depth - 1)
        events.append((match.end(), depth))
    return events


def _depth_at(events: list[tuple[int, int]], position: int) -> int:
    value = 0
    for event_position, depth in events:
        if event_position > position:
            break
        value = depth
    return value


def _matching_itemize_end(source: str, start: int, end: int, depth: int) -> int:
    current = depth
    for match in re.finditer(r"\\(?P<kind>begin|end)\s*\{itemize\}", source[start:end]):
        if match.group("kind") == "begin":
            current += 1
        else:
            current -= 1
            if current < depth:
                return start + match.start()
    return end


def _renumber(items: list[LatexItem]) -> list[LatexItem]:
    return [
        LatexItem(
            command=item.command,
            content_range=item.content_range,
            block_range=item.block_range,
            depth=item.depth,
            ordinal=index,
        )
        for index, item in enumerate(items, start=1)
    ]


def _is_commented(source: str, position: int) -> bool:
    line_start = source.rfind("\n", 0, position) + 1
    prefix = source[line_start:position]
    for match in re.finditer(r"%", prefix):
        slash_count = 0
        cursor = match.start() - 1
        while cursor >= 0 and prefix[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        if slash_count % 2 == 0:
            return True
    return False


def _line_start(source: str, position: int) -> int:
    return source.rfind("\n", 0, position) + 1
