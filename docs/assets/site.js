const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];

const form = $('[data-config-form]');
const output = $('#config-output code');
const note = $('[data-config-note]');

function field(name) {
  return $(`[data-field="${name}"]`, form);
}

function buildConfig() {
  const vocab = Number(field('vocab').value) || 50257;
  const hidden = Number(field('hidden').value) || 512;
  const layers = Number(field('layers').value) || 8;
  const heads = Number(field('heads').value) || 8;
  const sequence = Number(field('sequence').value) || 512;
  const attention = field('attention').value;
  const kvHeads = Number(field('kvHeads').value) || 2;
  const windowSize = field('window').value;
  const dataMode = field('dataMode').value;
  const useMoe = field('moe').checked;
  const nope = field('nope').checked;
  const compile = field('compile').checked;
  const isGqa = attention === 'gqa';
  const isMla = attention === 'mla';
  const windowLine = isMla || windowSize === 'none' ? '' : `    sliding_window_size: ${windowSize}\n`;
  const kvLine = isGqa ? `    num_key_value_heads: ${kvHeads}\n` : '';
  const mlaLine = isMla ? '    kv_lora_rank: 64\n' : '';
  const nopeLine = nope ? `    nope_layers: [${Math.max(1, Math.floor(layers / 3))}, ${Math.max(1, Math.floor(layers / 2))}]\n` : '';
  const moeBlock = useMoe ? '    moe_config:\n      num_experts: 4\n      top_k: 2\n      aux_loss_coefficient: 0.01\n' : '';
  let dataBlock;
  if (dataMode === 'hf') {
    dataBlock = `  tokenizer_name: gpt2\n  token_cache_dir: data/token_cache\n  datasets:\n    - hf_name: ${field('dataset').value || 'HuggingFaceFW/fineweb-edu'}\n      target_tokens: 100000000\n      weight: 1.0`;
  } else if (dataMode === 'cache') {
    dataBlock = '  dataset_token_path: data/tokens.uint16';
  } else {
    dataBlock = '  # No token cache: the CLI uses synthetic smoke data.';
  }
  const warning = [];
  if (hidden % heads !== 0) warning.push('hidden_size must be divisible by num_attention_heads.');
  if (isGqa && (kvHeads >= heads || heads % kvHeads !== 0)) warning.push('GQA requires KV heads smaller than, and dividing, query heads.');
  if (isMla && windowSize !== 'none') warning.push('MLA cannot be combined with sliding-window attention; the preview omits the window.');
  note.textContent = warning.length ? `Validation note: ${warning.join(' ')}` : 'Preview uses the strict nested Picotron configuration format.';
  note.classList.toggle('warning', warning.length > 0);
  output.textContent = `checkpoints:\n  checkpoint_interval: 500\n  checkpoints_path: checkpoints/demo.safetensors\n\nmodel:\n  dtype: auto\n  compile_model: ${compile}\n  triton_kernels:\n    rmsnorm: false\n    swiglu: false\n  model_config:\n    vocab_size: ${vocab}\n    hidden_size: ${hidden}\n    intermediate_size: ${hidden * 4}\n    num_hidden_layers: ${layers}\n    num_attention_heads: ${heads}\n    attention_type: ${attention}\n${kvLine}${mlaLine}    rope_theta: 1000000.0\n${nopeLine}${windowLine}${moeBlock}    tie_word_embeddings: true\n\noptimizer:\n  learning_rate_scheduler:\n    learning_rate: 0.0003\n  weight_decay: 0.1\n\nparallelism:\n  dp: 1\n  zero_stage: 0\n\ntokens:\n  sequence_length: ${sequence}\n  micro_batch_size: 2\n  train_steps: 1000\n\ndata:\n  vocab_size: ${vocab}\n${dataBlock}\n  num_workers: 4\n  prefetch_factor: 2\n\nlogging:\n  iteration_step_info_interval: 10\n  file_logging: true\n\ngeneral:\n  project: picotron\n  run: demo\n  seed: 1337`;
  $('[data-kv-control]').hidden = !isGqa;
  $('[data-hf-control]').hidden = dataMode !== 'hf';
}

form.addEventListener('input', buildConfig);
form.addEventListener('change', buildConfig);
buildConfig();

function showToast(message = 'Copied to clipboard') {
  const toast = $('[data-toast]');
  toast.textContent = message;
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 1800);
}

$$('[data-copy]').forEach((button) => button.addEventListener('click', async () => {
  await navigator.clipboard.writeText($(`#${button.dataset.copy}`).innerText);
  showToast();
}));

$$('[data-copy-text]').forEach((button) => button.addEventListener('click', async () => {
  await navigator.clipboard.writeText(button.dataset.copyText);
  showToast();
}));

$$('[data-tab]').forEach((button) => button.addEventListener('click', () => {
  $$('[data-tab]').forEach((item) => { item.classList.toggle('active', item === button); item.setAttribute('aria-selected', String(item === button)); });
  $$('[data-panel]').forEach((panel) => panel.classList.toggle('hidden', panel.dataset.panel !== button.dataset.tab));
}));

$$('[data-filter]').forEach((button) => button.addEventListener('click', () => {
  $$('[data-filter]').forEach((item) => item.classList.toggle('active', item === button));
  $$('.status-card').forEach((card) => card.classList.toggle('hidden', button.dataset.filter !== 'all' && card.dataset.status !== button.dataset.filter));
}));

const menuButton = $('[data-menu-button]');
const menu = $('[data-menu]');
menuButton.addEventListener('click', () => {
  const open = menu.classList.toggle('open');
  menuButton.setAttribute('aria-expanded', String(open));
});
$$('a', menu).forEach((link) => link.addEventListener('click', () => { menu.classList.remove('open'); menuButton.setAttribute('aria-expanded', 'false'); }));
