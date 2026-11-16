-- ==============================================================================
-- SentinelFL — Federated IoT Threat Intelligence
-- Migration: 00_base_schema.sql   (run FIRST on a new Supabase project)
-- Description:
--   Creates the base tables that migrations 01–04 secure and extend. Those scripts
--   only ALTER / add RLS policies and assume these tables already exist, so on a fresh
--   project they fail with "relation ... does not exist" without this file.
--
--   Columns are taken from what the application actually reads and writes
--   (frontend/src/lib/*.ts, frontend/src/contexts/*.tsx, backend/app/**). Migration 02
--   adds further optional dataset metadata columns with ADD COLUMN IF NOT EXISTS.
--
--   Idempotent: safe to run more than once.
-- ==============================================================================

BEGIN;

-- ------------------------------------------------------------------------------
-- profiles: display name per auth user (AuthContext reads/upserts full_name)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.profiles (
  id          uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name   text,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "profiles_select_own" ON public.profiles;
CREATE POLICY "profiles_select_own" ON public.profiles
  FOR SELECT TO authenticated USING (id = auth.uid());
DROP POLICY IF EXISTS "profiles_insert_own" ON public.profiles;
CREATE POLICY "profiles_insert_own" ON public.profiles
  FOR INSERT TO authenticated WITH CHECK (id = auth.uid());
DROP POLICY IF EXISTS "profiles_update_own" ON public.profiles;
CREATE POLICY "profiles_update_own" ON public.profiles
  FOR UPDATE TO authenticated USING (id = auth.uid()) WITH CHECK (id = auth.uid());

-- ------------------------------------------------------------------------------
-- workspaces + workspace_members (create_workspace RPC in 01 inserts into both)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.workspaces (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text NOT NULL,
  is_personal boolean NOT NULL DEFAULT false,
  owner_id    uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_workspaces_owner_id ON public.workspaces(owner_id);

CREATE TABLE IF NOT EXISTS public.workspace_members (
  workspace_id uuid NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
  user_id      uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role         text NOT NULL DEFAULT 'member',
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (workspace_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_workspace_members_user_id ON public.workspace_members(user_id);

-- ------------------------------------------------------------------------------
-- datasets: uploaded flow captures (datasetService.ts upserts id, name, storage_path,
-- workspace_id, status, uploaded_by, description[JSON metadata])
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.datasets (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id    uuid NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
  name            text NOT NULL,
  description     text,
  storage_path    text,
  status          text NOT NULL DEFAULT 'ready',
  uploaded_by     uuid REFERENCES auth.users(id) ON DELETE SET NULL,
  file_size_bytes bigint,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_datasets_workspace_id ON public.datasets(workspace_id);
CREATE INDEX IF NOT EXISTS idx_datasets_uploaded_by ON public.datasets(uploaded_by);

-- ------------------------------------------------------------------------------
-- preprocessing_jobs: backend job_runner inserts (id, dataset_id, status, result)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.preprocessing_jobs (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id          uuid NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
  workspace_id        uuid REFERENCES public.workspaces(id) ON DELETE CASCADE,
  status              text NOT NULL DEFAULT 'queued',
  target_column       text,
  scaler_type         text,
  imputation_strategy text,
  split_ratios        jsonb,
  feature_count       integer,
  result              jsonb,
  created_by          uuid REFERENCES auth.users(id) ON DELETE SET NULL DEFAULT auth.uid(),
  created_at          timestamptz NOT NULL DEFAULT now(),
  updated_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_preprocessing_jobs_dataset_id ON public.preprocessing_jobs(dataset_id);

-- ------------------------------------------------------------------------------
-- model_runs: training/evaluation records (modelService.ts joins datasets; 04 adds RLS)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.model_runs (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id  uuid NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,
  name        text,
  model_type  text,
  status      text NOT NULL DEFAULT 'completed',
  config      jsonb,
  metrics     jsonb,
  created_by  uuid REFERENCES auth.users(id) ON DELETE SET NULL DEFAULT auth.uid(),
  created_at  timestamptz NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- federated_rounds: per-round FL telemetry (WorkspaceContext counts these)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.federated_rounds (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id    uuid REFERENCES public.datasets(id) ON DELETE CASCADE,
  workspace_id  uuid REFERENCES public.workspaces(id) ON DELETE CASCADE,
  model_run_id  uuid REFERENCES public.model_runs(id) ON DELETE CASCADE,
  round_number  integer NOT NULL DEFAULT 0,
  metrics       jsonb,
  created_by    uuid REFERENCES auth.users(id) ON DELETE SET NULL DEFAULT auth.uid(),
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_federated_rounds_dataset_id ON public.federated_rounds(dataset_id);

-- Table privileges (Supabase grants these by default; repeated here for clarity).
-- Row access is still governed by the RLS policies in 00–04.
GRANT SELECT, INSERT, UPDATE, DELETE ON
  public.profiles, public.workspaces, public.workspace_members, public.datasets,
  public.preprocessing_jobs, public.model_runs, public.federated_rounds
TO authenticated;
GRANT ALL ON
  public.profiles, public.workspaces, public.workspace_members, public.datasets,
  public.preprocessing_jobs, public.model_runs, public.federated_rounds
TO service_role;

COMMIT;
