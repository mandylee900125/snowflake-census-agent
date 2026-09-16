-- Least-privilege role for the deployed app. Run once as ACCOUNTADMIN in
-- Snowsight, then set SNOWFLAKE_ROLE=CENSUS_READER in the app's secrets.
--
-- The role can use the warehouse and read the Marketplace share. It cannot
-- create, alter, or write anything, and it cannot call account-level
-- SYSTEM$ functions with effect. This is the last of four guard layers (see
-- README): if the topic gate, the SQL validator, and the execution limits
-- all failed, the session itself still refuses to write.

USE ROLE ACCOUNTADMIN;

CREATE ROLE IF NOT EXISTS CENSUS_READER
  COMMENT = 'Read-only access to the US Open Census share for the chat agent';

GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE CENSUS_READER;

-- Shared (Marketplace) databases are granted as a unit.
GRANT IMPORTED PRIVILEGES ON DATABASE US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET
  TO ROLE CENSUS_READER;

-- The login the app uses must hold the role. Replace with the app's user.
GRANT ROLE CENSUS_READER TO USER IDENTIFIER($APP_USER);

-- Verify: with the new role, a write must fail and a read must succeed.
-- USE ROLE CENSUS_READER;
-- CREATE TABLE should_fail (x INT);                       -- expect: insufficient privileges
-- SELECT COUNT(*) FROM "2020_CBG_B01";                    -- expect: 242335
