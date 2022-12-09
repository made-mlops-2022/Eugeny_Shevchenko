export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
export LOCAL_DATA_DIR=${PWD}/data
export LOCAL_MLRUNS_DIR=${PWD}/mlruns
export LOCAL_METRICS_DIR=${PWD}/metrics
# shellcheck disable=SC2155
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build