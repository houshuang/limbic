"""Small, non-authoritative index of the current queue for coordinator prompts."""
import re
from pathlib import Path

LIMIT = 4096
ITEM = re.compile(r'^- \[([ ~!xX])\] (.*)$')


def queue_items(text):
    items = []
    for line in text.splitlines():
        match = ITEM.match(line)
        if match:
            items.append([match[1], match[2]])
        elif items and line[:1] in (' ', '\t') and line.strip():
            items[-1][1] += ' ' + line.strip()
    return items


def summarize(mark, body):
    title = re.search(r'\*\*(.*?)\*\*', body)
    if title:
        name = title.group(1)
    else:
        name = body.split(':', 1)[0].split(' → ', 1)[0]
    # Preserve the assigned arrow owner, not every lane mentioned in history.
    owner = re.search(r'→\s*(?:lane|running driver|driver)\s+([\w.-]+)', body)
    if not owner:
        owner = re.search(r'\blane\s+([\w-]*\d[\w-]*)', body)
    waits = re.search(r'\bwaits? (?:on|until|for)\s+(.+?)(?=;|\.(?:\s|$)|$)', body, re.I)
    who = owner.group(1).rstrip('.') if owner else '?'
    return (f'- [{mark}] **{name}** · owner: {who}'
            + f' · wait: {waits.group(1).strip() if waits else "?"}')


def make_card(queue_path, incident_path):
    queue_path, incident_path = Path(queue_path), Path(incident_path)
    priority = ''
    try:
        source = queue_path.read_text()
        items = [summarize(mark, body) for mark, body in queue_items(source) if mark.lower() != 'x']
        queue = '\n'.join(items) if items else '(no open items found; check source if format changed)'
        lead = re.search(r'^\*\*Priority[^\n]*', source, re.M)
        priority = lead.group(0) + '\n' if lead else ''
    except (OSError, UnicodeError):
        items = []
        queue = f'(queue unavailable: {queue_path}; consult source before dispatch)'
    try:
        incident_text = incident_path.read_text()
        headings = [line.strip() for line in incident_text.splitlines() if re.match(r'^#{1,6}\s+', line)]
        incidents = '\n'.join(headings[-3:]) or '(no incident headings found)'
    except (OSError, UnicodeError):
        incidents = f'(incidents unavailable: {incident_path})'
    header = ('COORDINATOR CARD (index, not authority): Explicit user instructions and applicable repository rules prevail.\n'
              'Queue-listed owners/statuses may be stale or superseded; reconcile before action. ? = consult item. '
              'Read applicable AGENTS.md and perform mandatory startup checks. '
              'Use this card for routine orientation; consult the specific source item and event-relevant rules before action. '
              'If instructions require a file, read it; this card does not waive them.\n'
              f'Queue: {queue_path}\n')
    footer = f'\nRecent incident headings ({incident_path}):\n{incidents}\n'
    card = header + priority + queue + footer
    if len(card.encode('utf-8')) > LIMIT:
        # Keep the owner's priority and the first items in queue order; say how many more the source holds.
        kept = []
        for item in items:
            rest = len(items) - len(kept) - 1
            tail = f'(+{rest} more open items in the queue source)\n' if rest else ''
            trial = header + priority + '\n'.join(kept + [item]) + '\n' + tail + footer
            if len(trial.encode('utf-8')) > LIMIT:
                break
            kept.append(item)
        more = len(items) - len(kept)
        card = (header + priority + '\n'.join(kept) + ('\n' if kept else '')
                + (f'(+{more} more open items in the queue source)\n' if more else '') + footer)
    if len(card.encode('utf-8')) > LIMIT:
        card = ('COORDINATOR CARD: Explicit user instructions and applicable rules prevail. '
                'Card sources exceed limit; read the configured queue and incidents before dispatch.\n')
    return card
