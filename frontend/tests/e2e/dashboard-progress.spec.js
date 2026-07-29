import { expect, test } from '@playwright/test';

const DB_CALL_ID = '7dd91f4f-e089-4b2f-aea8-e93b838c20ad';
const PUBLIC_CALL_ID = 'en_CA_Banking_1586889';

function pipeline(currentId, percent) {
  const ids = [
    ['queued', 'Queued'],
    ['transcript', 'Transcript'],
    ['segments', 'Sentence segments'],
    ['acoustic', 'Audio signals'],
    ['evaluation', 'Evaluation context'],
    ['evidence', 'Prepare evidence'],
    ['requirements', 'Assess requirements'],
    ['findings', 'Ground findings'],
    ['decision', 'Set disposition'],
    ['presentation', 'Prepare evaluator'],
    ['publish', 'Publish results'],
    ['ready', 'Ready'],
  ];
  const currentIndex = ids.findIndex(([id]) => id === currentId);
  return {
    raw_stage: currentId,
    current_stage_id: currentId,
    current_stage_label: ids[currentIndex][1],
    percent,
    stages: ids.map(([id, label], index) => ({
      id,
      label,
      state: id === 'acoustic'
        ? 'skipped'
        : index < currentIndex
          ? 'completed'
          : index === currentIndex
            ? 'active'
            : 'pending',
    })),
  };
}

function catalogCall(overrides = {}) {
  return {
    call_id: PUBLIC_CALL_ID,
    db_call_id: DB_CALL_ID,
    domain: 'banking',
    accent: 'en-CA',
    duration: 636,
    analyzed: false,
    evaluation_available: false,
    evaluation_supported: true,
    evaluation_state: 'not_evaluated',
    evaluation_status: null,
    attention_required: null,
    checklist_counts: {
      demonstrated: 0,
      incorrect: 0,
      not_demonstrated: 0,
      unable_to_determine: 0,
    },
    checklist_total: 0,
    acoustic_status: null,
    acoustic_coverage: null,
    status: 'processing',
    stage: 'evaluating_v2_requirements',
    error: null,
    job_id: '75037e91-c17b-4827-8669-55639e24f56f',
    pipeline: pipeline('requirements', 54),
    ...overrides,
  };
}

test('shows real ordered worker progress in a dedicated activity section', async ({
  page,
}) => {
  await page.route('**/calls/catalog', (route) =>
    route.fulfill({ json: [catalogCall()] }),
  );
  await page.route(`**/calls/${DB_CALL_ID}/status`, (route) =>
    route.fulfill({
      json: {
        call_id: DB_CALL_ID,
        status: 'processing',
        stage: 'evaluating_v2_requirements',
        pipeline: pipeline('requirements', 54),
        error: null,
      },
    }),
  );

  await page.goto('/');

  const activity = page.getByRole('region', { name: 'Pipeline activity' });
  await expect(activity).toBeVisible();
  await expect(activity.getByText(PUBLIC_CALL_ID)).toBeVisible();
  await expect(
    activity.getByText('Assess requirements', { exact: true }).first(),
  ).toBeVisible();
  await expect(
    activity.getByText('Audio signals', { exact: true }),
  ).toBeVisible();
  await expect(activity.getByText('skipped', { exact: true })).toHaveCount(1);
  await expect(activity.getByText('54%', { exact: true })).toBeVisible();
});

test('allows a completed call to be deliberately run again', async ({
  page,
}) => {
  await page.route('**/calls/catalog', (route) =>
    route.fulfill({
      json: [
        catalogCall({
          analyzed: true,
          evaluation_available: true,
          evaluation_state: 'no_attention_finding',
          evaluation_status: 'complete',
          attention_required: false,
          checklist_counts: {
            demonstrated: 8,
            incorrect: 0,
            not_demonstrated: 0,
            unable_to_determine: 0,
          },
          checklist_total: 8,
          acoustic_status: 'limited',
          acoustic_coverage: 'Audio support on 71 of 152 segments',
          status: 'succeeded',
          stage: 'done',
          pipeline: pipeline('ready', 100),
        }),
        catalogCall({
          call_id: 'en_CA_Health_1587315',
          db_call_id: 'af66f0cc-8281-4f99-9a6e-53c1ac8ae936',
          domain: 'health',
          evaluation_supported: false,
          evaluation_state: 'unsupported',
          status: 'available',
          stage: 'uploaded',
          pipeline: null,
        }),
      ],
    }),
  );
  await page.route('**/calls/analyze', async (route) => {
    const payload = route.request().postDataJSON();
    expect(payload).toEqual({
      call_ids: [DB_CALL_ID],
      force: true,
    });
    await route.fulfill({
      status: 202,
      json: {
        jobs: [{
          call_id: DB_CALL_ID,
          public_call_id: PUBLIC_CALL_ID,
          job_id: '0ce6b173-e09b-423a-a256-c634cfdb21bd',
          status: 'queued',
          created: true,
        }],
      },
    });
  });
  await page.route(`**/calls/${DB_CALL_ID}/status`, (route) =>
    route.fulfill({
      json: {
        call_id: DB_CALL_ID,
        status: 'processing',
        stage: 'transcribing',
        pipeline: pipeline('transcript', 14),
        error: null,
      },
    }),
  );

  await page.goto('/');
  await expect(page.locator('tbody tr').first()).toContainText(PUBLIC_CALL_ID);
  await page.getByRole('checkbox', {
    name: `Select ${PUBLIC_CALL_ID}`,
  }).check();
  await page.getByRole('button', { name: 'Analyze 1 call' }).click();

  await expect(
    page.getByRole('region', { name: 'Pipeline activity' }),
  ).toBeVisible();
});
