import { expect, test } from '@playwright/test';

const DB_CALL_ID = '7dd91f4f-e089-4b2f-aea8-e93b838c20ad';
const PUBLIC_CALL_ID = 'en_CA_Banking_1586889';

function pipeline(currentId, percent) {
  const ids = [
    ['queued', 'Queued'],
    ['transcript', 'Transcript'],
    ['segments', 'Sentence segments'],
    ['acoustic', 'Audio signals'],
    ['evaluation', 'LLM evaluation'],
    ['evaluation_v2', 'V2 shadow'],
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
      state: id === 'acoustic' || id === 'evaluation_v2'
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
    status: 'processing',
    stage: 'evaluating',
    error: null,
    job_id: '75037e91-c17b-4827-8669-55639e24f56f',
    pipeline: pipeline('evaluation', 50),
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
        stage: 'evaluating',
        pipeline: pipeline('evaluation', 50),
        error: null,
      },
    }),
  );

  await page.goto('/');

  const activity = page.getByRole('region', { name: 'Pipeline activity' });
  await expect(activity).toBeVisible();
  await expect(activity.getByText(PUBLIC_CALL_ID)).toBeVisible();
  await expect(
    activity.getByText('LLM evaluation', { exact: true }).first(),
  ).toBeVisible();
  await expect(
    activity.getByText('Audio signals', { exact: true }),
  ).toBeVisible();
  await expect(activity.getByText('skipped', { exact: true })).toHaveCount(2);
  await expect(activity.getByText('50%', { exact: true })).toBeVisible();
});

test('allows a completed call to be deliberately run again', async ({
  page,
}) => {
  await page.route('**/calls/catalog', (route) =>
    route.fulfill({
      json: [
        catalogCall({
          analyzed: true,
          status: 'succeeded',
          stage: 'done',
          pipeline: pipeline('ready', 100),
          compliance_passed: 5,
          compliance_applicable: 6,
          workflow_met: 8,
          workflow_total: 8,
          risk_level: 'none',
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
  await page.getByRole('checkbox', {
    name: `Select ${PUBLIC_CALL_ID}`,
  }).check();
  await page.getByRole('button', { name: 'Analyze 1 call' }).click();

  await expect(
    page.getByRole('region', { name: 'Pipeline activity' }),
  ).toBeVisible();
});
