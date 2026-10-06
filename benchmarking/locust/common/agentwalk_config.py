# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent-walk benchmark runtime flags.

Defaults here mirror the Go side (internal/benchmarking/boomer/agentwalk/
walk.go) so a run with no flags behaves the same whether the values reach
the worker through /boomer-config or through --config-json.
"""

from locust import events
from locust.argument_parser import LocustArgumentParser


@events.init_command_line_parser.add_listener
def add_agentwalk_arguments(parser: LocustArgumentParser) -> None:
    parser.add_argument(
        "--agentwalk-cron-interval",
        type=float,
        default=1800.0,
        env_var="LOCUST_AGENTWALK_CRON_INTERVAL",
        help=(
            "Seconds between an agent's wakeup ticks. Each agent keeps its own "
            "phase, and a tick that falls while a session is running is "
            "skipped (default: 1800, 30 minutes)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-idle-cpu",
        type=float,
        default=0.25,
        env_var="LOCUST_AGENTWALK_IDLE_CPU",
        help=(
            "Fraction of one core an awake agent burns while idle, applied once "
            "per actor through glutton UseCPU; 0 disables (default: 0.25)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-resident-ram",
        type=str,
        default="64Mi",
        env_var="LOCUST_AGENTWALK_RESIDENT_RAM",
        help=(
            "Memory each agent holds from creation on, as a Kubernetes quantity: "
            "the heap a real agent process keeps whether or not it is busy. In "
            "every snapshot; the long operations churn within it (default: 64Mi)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-weight-llm",
        type=float,
        default=50.0,
        env_var="LOCUST_AGENTWALK_WEIGHT_LLM",
        help=(
            "Weight of an LLM query when an active agent picks its next action: "
            "the actor is suspended for a log-normal think time (default: 50)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-weight-short",
        type=float,
        default=30.0,
        env_var="LOCUST_AGENTWALK_WEIGHT_SHORT",
        help="Weight of a short on-sandbox CPU burst, no suspend (default: 30)",
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-weight-long",
        type=float,
        default=10.0,
        env_var="LOCUST_AGENTWALK_WEIGHT_LONG",
        help=(
            "Weight of a long on-sandbox operation: about a minute of CPU, "
            "disk, or RAM work, no suspend (default: 10)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-weight-done",
        type=float,
        default=10.0,
        env_var="LOCUST_AGENTWALK_WEIGHT_DONE",
        help=(
            "Weight of ending the session. Never picked first, nor before an "
            "LLM query (default: 10)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-think-mu",
        type=float,
        default=2.0414,
        env_var="LOCUST_AGENTWALK_THINK_MU",
        help=(
            "Log-space mean of the log-normal LLM think time, in seconds; the "
            "SWE-perf default puts the median near 7.7s (default: 2.0414)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-think-sigma",
        type=float,
        default=0.8674,
        env_var="LOCUST_AGENTWALK_THINK_SIGMA",
        help=(
            "Log-space standard deviation of the LLM think time; the SWE-perf "
            "default puts the 90th percentile near 23.4s (default: 0.8674)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-short-seconds",
        type=float,
        default=6.0,
        env_var="LOCUST_AGENTWALK_SHORT_SECONDS",
        help="Mean length of a short compute burst; each draw is within ±50% (default: 6)",
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-long-seconds",
        type=float,
        default=60.0,
        env_var="LOCUST_AGENTWALK_LONG_SECONDS",
        help="Mean length of a long operation; each draw is within ±50% (default: 60)",
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-short-max-cores",
        type=int,
        default=1,
        env_var="LOCUST_AGENTWALK_SHORT_MAX_CORES",
        help="Ceiling on a short burst's goroutines; each burst draws from 1 to this (default: 1)",
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-long-max-cores",
        type=int,
        default=2,
        env_var="LOCUST_AGENTWALK_LONG_MAX_CORES",
        help="Ceiling on a long operation's CPU goroutines; each operation draws from 1 to this (default: 2)",
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-long-max-disk",
        type=str,
        default="16Mi",
        env_var="LOCUST_AGENTWALK_LONG_MAX_DISK",
        help=(
            "Ceiling on the bytes a long operation writes and reads back per cycle, "
            "as a Kubernetes quantity; the operation's disk share scales it. The data "
            "dir is tmpfs, so this is memory too (default: 16Mi)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-long-max-ram",
        type=str,
        default="32Mi",
        env_var="LOCUST_AGENTWALK_LONG_MAX_RAM",
        help=(
            "Ceiling on the bytes a long operation churns and walks per cycle, as a "
            "Kubernetes quantity, and the resident working set every actor ends up "
            "holding (default: 32Mi)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-long-cycle-seconds",
        type=float,
        default=2.0,
        env_var="LOCUST_AGENTWALK_LONG_CYCLE_SECONDS",
        help=(
            "Nominal length of one cycle of a long operation; the CPU and idle shares "
            "are fractions of it, and the operation repeats cycles until its length is "
            "up (default: 2)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-long-jitter",
        type=float,
        default=0.2,
        env_var="LOCUST_AGENTWALK_LONG_JITTER",
        help=(
            "How far each cycle's intensities stray from the operation's draw, as a "
            "fraction in [0, 1) (default: 0.2)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-max-actions",
        type=int,
        default=50,
        env_var="LOCUST_AGENTWALK_MAX_ACTIONS",
        help=(
            "Actions after which a session is wound up regardless of the "
            "weights, with one LLM query first if none has happened (default: 50)"
        ),
        include_in_web_ui=True,
    )
    parser.add_argument(
        "--agentwalk-template",
        type=str,
        default="glutton",
        env_var="LOCUST_AGENTWALK_TEMPLATE",
        help="ActorTemplate in benchmark-workloads each agent is created from (default: glutton)",
        include_in_web_ui=True,
    )
