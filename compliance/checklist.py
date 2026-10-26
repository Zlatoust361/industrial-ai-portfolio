"""Structured ИБ acceptance checklist (week 21).

Extends the EXISTING opросник from
`course/modules/ot-integration-for-cv-engineer/day02-ot-it-network-
segmentation.md`'s section 4 ("Что именно проверяет служба ИБ на
приёмке") -- not a new checklist from scratch -- plus every item from
`course/TECH-DEBT.md`'s section F ("Эксплуатационные и security-риски").
Every item below is a REAL, already-documented finding against THIS
repository's actual code/docs, not invented for this week.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChecklistStatus(str, Enum):  # noqa: UP042 -- py3.10 compat, see src/config/environment.py::Environment
    """Whether one checklist item is resolved, still open, or doesn't apply."""

    CLOSED = "closed"
    OPEN = "open"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ChecklistItem:
    """One acceptance-checklist item.

    Attributes:
        item_id: Short stable identifier (`"OT-1"`, `"SEC-3"`, ...).
        description: What's being checked.
        status: Current status -- see `ChecklistStatus`.
        justification: Why this status, in this repository, today --
            never a bare status with no reasoning.
        source: Where this item comes from (a real doc section, not a
            made-up reference).
    """

    item_id: str
    description: str
    status: ChecklistStatus
    justification: str
    source: str


_OT_SOURCE = "course/modules/ot-integration-for-cv-engineer/day02-ot-it-network-segmentation.md#4"
_SEC_SOURCE = "course/TECH-DEBT.md#f-эксплуатационные-и-security-риски"
_OPEN_PHYSICAL_CHECK = (
    "Требует реального устройства/сети на площадке -- не проверяется в этой песочнице."
)

DEFAULT_CHECKLIST: tuple[ChecklistItem, ...] = (
    # From the OT-integration module's day 2 опросник -- a PHYSICAL
    # deployment checklist, so every item here defaults to OPEN ("needs
    # verification against a real device/network") rather than CLOSED --
    # a real pilot fills these in per site; this module only carries
    # the structure, not a fabricated "done".
    ChecklistItem(
        item_id="OT-1",
        description=(
            "Список всех сетевых подключений Jetson (исходящих и входящих) "
            "с портами и протоколами."
        ),
        status=ChecklistStatus.OPEN,
        justification=_OPEN_PHYSICAL_CHECK,
        source=_OT_SOURCE,
    ),
    ChecklistItem(
        item_id="OT-2",
        description="Отсутствие дефолтных/захардкоженных паролей в контейнерах и на самом Jetson.",
        status=ChecklistStatus.OPEN,
        justification=(
            f"{_OPEN_PHYSICAL_CHECK} "
            "См. также SEC-2 ниже про известный дефолтный пароль Grafana в этом репозитории."
        ),
        source=_OT_SOURCE,
    ),
    ChecklistItem(
        item_id="OT-3",
        description=(
            "Отключены неиспользуемые сервисы (SSH только с ключом, никакого Telnet/FTP, "
            "mDNS/UPnP отключены при отсутствии нужды)."
        ),
        status=ChecklistStatus.OPEN,
        justification=_OPEN_PHYSICAL_CHECK,
        source=_OT_SOURCE,
    ),
    ChecklistItem(
        item_id="OT-4",
        description=(
            "Обновления JetPack/пакетов -- управляемый процесс, "
            "а не автоматический apt upgrade по расписанию."
        ),
        status=ChecklistStatus.OPEN,
        justification=_OPEN_PHYSICAL_CHECK,
        source=_OT_SOURCE,
    ),
    ChecklistItem(
        item_id="OT-5",
        description=(
            "Логи системы уходят в SIEM/централизованный лог-сервер предприятия, "
            "если такой есть."
        ),
        status=ChecklistStatus.OPEN,
        justification=_OPEN_PHYSICAL_CHECK,
        source=_OT_SOURCE,
    ),
    # From TECH-DEBT.md's section F -- real findings already documented
    # against this repository's actual code.
    ChecklistItem(
        item_id="SEC-1",
        description=(
            "Два параллельных пути деплоя vlm-inference (docker-compose SSH vs k3s) "
            "без защиты от одновременного запуска на тот же порт."
        ),
        status=ChecklistStatus.CLOSED,
        justification=(
            "scripts/deploy_to_jetson.sh проверяет по SSH 'kubectl get deployment' перед "
            "деплоем и явно отказывает (DeployPathConflictError), если k3s-путь уже активен. "
            "НЕ верифицировано против реального k3s-кластера (нет физического устройства в "
            "этой песочнице) -- проверено только локальным bash-харнессом с поддельным ssh."
        ),
        source=f"{_SEC_SOURCE} (пункт 1)",
    ),
    ChecklistItem(
        item_id="SEC-2",
        description=(
            "GF_SECURITY_ADMIN_PASSWORD: admin в docker-compose.monitoring.yml -- "
            "дефолтный пароль Grafana."
        ),
        status=ChecklistStatus.OPEN,
        justification=(
            "Задокументирован как 'смени перед реальным использованием' в monitoring/README.md, "
            "но ничего программно не форсирует смену -- нет проверки при старте, которая бы "
            "отказала с дефолтным паролем."
        ),
        source=f"{_SEC_SOURCE} (пункт 2)",
    ),
    ChecklistItem(
        item_id="SEC-3",
        description="k3s/'s replicas: 1 без PodDisruptionBudget/health-based traffic draining.",
        status=ChecklistStatus.NOT_APPLICABLE,
        justification=(
            "Осознанный выбор для однодевайсного пилота (см. TECH-DEBT.md раздел B.6) -- риск "
            "актуален только если манифесты скопируют как шаблон для многоузлового кластера, "
            "не читая явные оговорки k3s/README.md про 'рассчитано на один узел'."
        ),
        source=f"{_SEC_SOURCE} (пункт 3)",
    ),
    ChecklistItem(
        item_id="SEC-4",
        description=(
            "Self-hosted GitHub Actions раннер на устройстве, обрабатывающий push в main -- "
            "кто именно имеет право merge, не зафиксировано."
        ),
        status=ChecklistStatus.OPEN,
        justification=(
            "jetson-cicd.yml's deploy-job помечен environment: jetson-production (механизм "
            "'required reviewers' технически применим), но список reviewers администратор "
            "репозитория ещё не настроил в Settings -- само организационное решение не принято."
        ),
        source=f"{_SEC_SOURCE} (пункт 4)",
    ),
    ChecklistItem(
        item_id="SEC-5",
        description="ANTHROPIC_API_KEY никогда не прогонялся против настоящего https://api.anthropic.com.",
        status=ChecklistStatus.OPEN,
        justification=(
            "Весь код реален и протестирован через httpx.MockTransport, но реальные "
            "rate-limit'ы/задержки/форматы ошибок настоящего API могут отличаться -- первый "
            "реальный вызов на проде будет и первой реальной проверкой этого пути целиком."
        ),
        source=f"{_SEC_SOURCE} (пункт 5)",
    ),
    ChecklistItem(
        item_id="SEC-6",
        description=(
            "restart_training в assistant/action_executor.py не проверяет права вызывающего."
        ),
        status=ChecklistStatus.OPEN,
        justification=(
            "_handle_restart_training выполняет команду для любого вызывающего -- в коде нет "
            "понятия роли 'оператор' вообще; расходится с тем, что modules/operator-training/ "
            "обещает оператору (ссылается на несуществующий ADR в DECISIONS.md)."
        ),
        source=f"{_SEC_SOURCE} (пункт 6)",
    ),
)


def checklist_summary(items: tuple[ChecklistItem, ...] = DEFAULT_CHECKLIST) -> dict[str, int]:
    """Count items per `ChecklistStatus`.

    Args:
        items: Checklist items to summarize -- defaults to `DEFAULT_CHECKLIST`.

    Returns:
        `{"closed": N, "open": N, "not_applicable": N}` -- all three
        keys always present, `0` if unused.
    """
    counts = {status.value: 0 for status in ChecklistStatus}
    for item in items:
        counts[item.status.value] += 1
    return counts


def render_checklist_markdown(items: tuple[ChecklistItem, ...] = DEFAULT_CHECKLIST) -> str:
    """Render the checklist as a Markdown table, for the demo report."""
    lines = ["| ID | Статус | Описание | Обоснование | Источник |", "|---|---|---|---|---|"]
    for item in items:
        lines.append(
            f"| {item.item_id} | {item.status.value} | {item.description} | "
            f"{item.justification} | {item.source} |"
        )
    return "\n".join(lines)
