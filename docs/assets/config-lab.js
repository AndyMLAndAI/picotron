const select = (selector, scope = document) => scope.querySelector(selector);
const selectAll = (selector, scope = document) => [...scope.querySelectorAll(selector)];

const form = select('[data-lab-form]');
const datasets = select('[data-dataset-list]');
const yamlOutput = select('[data-lab-output]');
const validationNote = select('[data-lab-note]');

const datasetTemplate = (index) => `
  <article class="dataset-card" data-dataset-card>
    <div class="dataset-card-top"><strong>Dataset ${index + 1}</strong><button type="button" class="remove-dataset" data-remove-dataset>Remove</button></div>
    <div class="lab-fields lab-fields-3">
      <label>Source type <select data-dataset="type"><option value="hf">Hugging Face source</option><option value="path">Existing uint16 cache</option></select></label>
      <label data-source-label>HF dataset name <input data-dataset="source" type="text" value="HuggingFaceFW/fineweb-edu"></label>
      <label>Mix weight <input data-dataset="weight" type="number" min="0.0001" step="0.1" value="1.0"></label>
      <label data-hf-only>HF config <input data-dataset="hfConfig" type="text" value="sample-10BT" placeholder="optional"></label>
      <label data-hf-only>Target tokens <input data-dataset="targetTokens" type="number" min="1" step="1000000" value="100000000"></label>
      <label data-hf-only>Text field <input data-dataset="textField" type="text" value="text"></label>
    </div>
  </article>`;

function field(name) {
  return select(`[data-lab="${name}"]`, form);
}

function number(name, fallback) {
  return Number(field(name).value) || fallback;
}

function yamlQuote(value) {
  return JSON.stringify(String(value));
}

function noPeLayers(value) {
  if (!value.trim()) return [];
  return value.split(',').map((item) => Number(item.trim()));
}

function updateDatasetCard(card) {
  const isHf = select('[data-dataset="type"]', card).value === 'hf';
  select('[data-source-label]', card).firstChild.textContent = isHf ? 'HF dataset name ' : 'uint16 cache path ';
  selectAll('[data-hf-only]', card).forEach((item) => { item.hidden = !isHf; });
}

function datasetSpecs() {
  return selectAll('[data-dataset-card]').map((card) => ({
    type: select('[data-dataset="type"]', card).value,
    source: select('[data-dataset="source"]', card).value.trim(),
    weight: Number(select('[data-dataset="weight"]', card).value),
    hfConfig: select('[data-dataset="hfConfig"]', card).value.trim(),
    targetTokens: Number(select('[data-dataset="targetTokens"]', card).value),
    textField: select('[data-dataset="textField"]', card).value.trim() || 'text',
  }));
}

function buildYaml() {
  const vocab = number('vocab', 50257);
  const sequence = number('sequence', 512);
  const hidden = number('hidden', 512);
  const intermediate = number('intermediate', hidden * 4);
  const layers = number('layers', 8);
  const heads = number('heads', 8);
  const attention = field('attention').value;
  const kvHeads = number('kvHeads', 2);
  const windowSize = number('window', 256);
  const ropeTheta = number('ropeTheta', 1000000);
  const nope = noPeLayers(field('nope').value);
  const useMoe = field('moe').checked;
  const sources = datasetSpecs();
  const warnings = [];
  if (hidden % heads !== 0) warnings.push('hidden_size must divide evenly by num_attention_heads.');
  if (attention === 'gqa' && (kvHeads >= heads || heads % kvHeads !== 0)) warnings.push('GQA needs KV heads smaller than and dividing query heads.');
  if (attention === 'mla' && windowSize > 0) warnings.push('MLA cannot combine with sliding-window attention; the generated YAML omits the window.');
  if (nope.some((layer) => !Number.isInteger(layer) || layer < 0 || layer >= layers)) warnings.push('Every NoPE layer must be a valid zero-based layer index.');
  if (sources.length === 0) warnings.push('Add a dataset before launching pretraining; no source means the CLI uses synthetic smoke data.');
  const invalidSource = sources.some((source) => !source.source || !Number.isFinite(source.weight) || source.weight <= 0 || (source.type === 'hf' && (!Number.isInteger(source.targetTokens) || source.targetTokens <= 0)));
  if (invalidSource) warnings.push('Every dataset needs a non-empty source, positive weight, and positive target tokens for HF sources.');
  const kvLine = attention === 'gqa' ? `    num_key_value_heads: ${kvHeads}\n` : '';
  const mlaLine = attention === 'mla' ? `    kv_lora_rank: ${number('mlaRank', 64)}\n` : '';
  const windowLine = attention === 'mla' ? '' : `    sliding_window_size: ${windowSize}\n`;
  const nopeLine = nope.length ? `    nope_layers: [${nope.join(', ')}]\n` : '';
  const moeBlock = useMoe ? `    moe_config:\n      num_experts: ${number('experts', 4)}\n      top_k: ${number('topK', 2)}\n      aux_loss_coefficient: ${number('aux', 0.01)}\n` : '    moe_config: null\n';
  const dataSources = sources.map((source) => source.type === 'hf'
    ? `    - hf_name: ${yamlQuote(source.source)}\n${source.hfConfig ? `      hf_config: ${yamlQuote(source.hfConfig)}\n` : ''}      target_tokens: ${source.targetTokens}\n      text_field: ${yamlQuote(source.textField)}\n      weight: ${source.weight}`
    : `    - path: ${yamlQuote(source.source)}\n      weight: ${source.weight}`).join('\n');
  yamlOutput.textContent = `checkpoints:\n  checkpoint_interval: ${number('checkpointInterval', 500)}\n  checkpoints_path: checkpoints/${field('run').value.trim() || 'experiment'}.safetensors\n  save_final_state: true\n\nmodel:\n  dtype: ${field('dtype').value}\n  compile_model: ${field('compile').checked}\n  triton_kernels:\n    rmsnorm: false\n    swiglu: false\n  model_config:\n    vocab_size: ${vocab}\n    hidden_size: ${hidden}\n    intermediate_size: ${intermediate}\n    num_hidden_layers: ${layers}\n    num_attention_heads: ${heads}\n    attention_type: ${attention}\n${kvLine}${mlaLine}    rope_theta: ${ropeTheta}\n${nopeLine}${windowLine}${moeBlock}    tie_word_embeddings: true\n    position_embedding_type: rope\n    gradient_checkpointing: ${field('checkpointing').checked}\n\noptimizer:\n  learning_rate_scheduler:\n    learning_rate: ${number('learningRate', 0.0003)}\n  weight_decay: ${number('weightDecay', 0.1)}\n  clip_grad: 1.0\n\nparallelism:\n  dp: ${number('dp', 1)}\n  zero_stage: ${field('zero').value}\n\ntokens:\n  sequence_length: ${sequence}\n  micro_batch_size: ${number('batch', 2)}\n  train_steps: ${number('steps', 1000)}\n\ndata:\n  tokenizer_name: gpt2\n  vocab_size: ${vocab}\n  token_cache_dir: data/token_cache\n${sources.length ? `  datasets:\n${dataSources}\n` : ''}  num_workers: 4\n  prefetch_factor: 2\n\nlogging:\n  iteration_step_info_interval: 10\n  file_logging: true\n\ngeneral:\n  project: picotron\n  run: ${yamlQuote(field('run').value.trim() || 'experiment')}\n  seed: 1337`;
  validationNote.textContent = warnings.length ? `Review before use: ${warnings.join(' ')}` : 'Schema checks look consistent. Confirm GPU memory, dataset access, and throughput on your target hardware before a long run.';
  validationNote.classList.toggle('warning', warnings.length > 0);
  select('[data-lab-kv]').hidden = attention !== 'gqa';
  select('[data-lab-mla]').hidden = attention !== 'mla';
  select('[data-lab-moe]').hidden = !useMoe;
}

function addDataset() {
  datasets.insertAdjacentHTML('beforeend', datasetTemplate(selectAll('[data-dataset-card]').length));
  const card = datasets.lastElementChild;
  updateDatasetCard(card);
  card.addEventListener('input', buildYaml);
  card.addEventListener('change', () => { updateDatasetCard(card); buildYaml(); });
  select('[data-remove-dataset]', card).addEventListener('click', () => {
    card.remove();
    selectAll('[data-dataset-card]').forEach((item, index) => { select('strong', item).textContent = `Dataset ${index + 1}`; });
    buildYaml();
  });
}

select('[data-add-dataset]').addEventListener('click', addDataset);
form.addEventListener('input', buildYaml);
form.addEventListener('change', buildYaml);
select('[data-lab-copy]').addEventListener('click', async () => {
  await navigator.clipboard.writeText(yamlOutput.innerText);
  const toast = select('[data-toast]');
  toast.classList.add('show');
  window.setTimeout(() => toast.classList.remove('show'), 1800);
});

addDataset();
buildYaml();
