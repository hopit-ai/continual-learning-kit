# Source after binding PARENT_OUT, DURABLE_ROOT, KIT and TRAINER_PY.
# Every long command must run via run_stage so bash can handle TERM in wait.
: "${TERM_LEAD_SECONDS:=1800}" "${SAVE_ALLOWANCE_SECONDS:=900}"
if (( TERM_LEAD_SECONDS < SAVE_ALLOWANCE_SECONDS || SAVE_ALLOWANCE_SECONDS <= 0 )); then
  echo 'TERM lead must cover the positive incremental save allowance' >&2
  exit 2
fi
child=''
save_handoff() {
  "$TRAINER_PY" "$KIT/message_handoff.py" --out "$PARENT_OUT" --durable "$DURABLE_ROOT" --action save
}
stop_and_save() {
  trap '' TERM INT
  if [[ -n "$child" ]]; then
    kill -TERM -- "-$child" 2>/dev/null || :
    local deadline=$((SECONDS + TERM_LEAD_SECONDS - SAVE_ALLOWANCE_SECONDS))
    while kill -0 -- "-$child" 2>/dev/null && (( SECONDS < deadline )); do sleep 0.1; done
    if kill -0 -- "-$child" 2>/dev/null; then kill -KILL -- "-$child" 2>/dev/null || :; fi
    wait "$child" 2>/dev/null || :
    child=''
  fi
  exit 143  # EXIT saves after the writer has stopped.
}
run_stage() {
  local status=0 previous_options=$-
  # Bash monitor mode gives this stage its own process group, including workers.
  set -m
  "$@" & child=$!
  [[ "$previous_options" == *m* ]] || set +m
  wait "$child" || status=$?
  child=''
  return "$status"
}
trap save_handoff EXIT
trap stop_and_save TERM INT
