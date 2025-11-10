set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
  CREATE DATABASE lego;
  CREATE DATABASE titanic;
  CREATE DATABASE happiness;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "lego"      -f /docker-entrypoint-initdb.d/lego.sql
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "titanic"   -f /docker-entrypoint-initdb.d/titanic.sql
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "happiness" -f /docker-entrypoint-initdb.d/happiness_index.sql
