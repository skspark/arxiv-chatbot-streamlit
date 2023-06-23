#!/bin/zsh

REQUIRED_COMMANDS=( "pre-commit" "pylint" "pdm" "pytest" )

check_command_installed() {
  local cmd=$1
  if ! command -v $cmd &> /dev/null
  then
      echo "!! command \"$cmd\" not found. please install it"
      exit 1
  fi
}

# check required commands
for cmd in "${REQUIRED_COMMANDS[@]}"
do
    check_command_installed $cmd
done

pre-commit install
