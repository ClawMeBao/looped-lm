const fs = require('fs');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const { fileNodes, importEdges, allEdges } = input;

// A. Directory Grouping
const allPaths = fileNodes.map(n => n.filePath);
// find common prefix
function commonPrefix(paths) {
  if (!paths.length) return '';
  const parts = paths[0].split('/');
  let prefix = '';
  for (let i = 0; i < parts.length - 1; i++) {
    const seg = parts.slice(0, i+1).join('/') + '/';
    if (paths.every(p => p.startsWith(seg))) prefix = seg;
    else break;
  }
  return prefix;
}
const prefix = commonPrefix(allPaths);
const directoryGroups = {};
for (const node of fileNodes) {
  const rel = node.filePath.startsWith(prefix) ? node.filePath.slice(prefix.length) : node.filePath;
  const seg = rel.includes('/') ? rel.split('/')[0] : 'root';
  if (!directoryGroups[seg]) directoryGroups[seg] = [];
  directoryGroups[seg].push(node.id);
}

// B. Node Type Grouping
const nodeTypeGroups = {};
for (const node of fileNodes) {
  if (!nodeTypeGroups[node.type]) nodeTypeGroups[node.type] = [];
  nodeTypeGroups[node.type].push(node.id);
}

// C. Fan-in / Fan-out
const fanIn = {}, fanOut = {};
for (const node of fileNodes) { fanIn[node.id] = 0; fanOut[node.id] = 0; }
for (const e of importEdges) {
  if (fanOut[e.source] !== undefined) fanOut[e.source]++;
  if (fanIn[e.target] !== undefined) fanIn[e.target]++;
}

// D. Cross-category edges
const ccMap = {};
for (const e of allEdges) {
  const srcNode = fileNodes.find(n => n.id === e.source);
  const tgtNode = fileNodes.find(n => n.id === e.target);
  if (!srcNode || !tgtNode) continue;
  const key = `${srcNode.type}|${tgtNode.type}|${e.type}`;
  ccMap[key] = (ccMap[key] || 0) + 1;
}
const crossCategoryEdges = Object.entries(ccMap).map(([k,count]) => {
  const [fromType, toType, edgeType] = k.split('|');
  return { fromType, toType, edgeType, count };
});

// E. Inter-group imports
const nodeToGroup = {};
for (const [grp, ids] of Object.entries(directoryGroups)) for (const id of ids) nodeToGroup[id] = grp;
const interMap = {};
for (const e of importEdges) {
  const sg = nodeToGroup[e.source], tg = nodeToGroup[e.target];
  if (!sg || !tg || sg === tg) continue;
  const k = `${sg}|${tg}`;
  interMap[k] = (interMap[k] || 0) + 1;
}
const interGroupImports = Object.entries(interMap).map(([k,count]) => {
  const [from,to] = k.split('|'); return {from,to,count};
}).sort((a,b) => b.count - a.count);

// F. Intra-group density
const intraGroupDensity = {};
for (const grp of Object.keys(directoryGroups)) {
  const ids = new Set(directoryGroups[grp]);
  let internal = 0, total = 0;
  for (const e of importEdges) {
    const hasS = ids.has(e.source), hasT = ids.has(e.target);
    if (hasS || hasT) total++;
    if (hasS && hasT) internal++;
  }
  intraGroupDensity[grp] = { internalEdges: internal, totalEdges: total, density: total ? internal/total : 0 };
}

// G. Pattern matching
const dirPatterns = {
  routes:'api', api:'api', controllers:'api', endpoints:'api', handlers:'api',
  services:'service', core:'service', lib:'service', domain:'service', logic:'service',
  models:'data', db:'data', data:'data', persistence:'data', repository:'data', entities:'data',
  components:'ui', views:'ui', pages:'ui', ui:'ui', layouts:'ui', screens:'ui',
  middleware:'middleware', plugins:'middleware', interceptors:'middleware', guards:'middleware',
  utils:'utility', helpers:'utility', common:'utility', shared:'utility', tools:'utility',
  config:'config', constants:'config', env:'config', settings:'config',
  __tests__:'test', test:'test', tests:'test', spec:'test', specs:'test',
  types:'types', interfaces:'types', schemas:'types', contracts:'types', dtos:'types',
  hooks:'hooks', store:'state', state:'state', reducers:'state', actions:'state', slices:'state',
  assets:'assets', static:'assets', public:'assets', migrations:'data',
  docs:'documentation', documentation:'documentation', wiki:'documentation',
  deploy:'infrastructure', infra:'infrastructure', infrastructure:'infrastructure',
  scripts:'entry', src:'service', phase0:'service', phase1:'service', phase2:'service',
  root:'config'
};
const patternMatches = {};
for (const grp of Object.keys(directoryGroups)) {
  patternMatches[grp] = dirPatterns[grp] || 'utility';
}

// H. Deployment topology
const infraExts = ['.dockerfile','docker','docker-compose','makefile','.tf','.tfvars'];
const infraFiles = fileNodes.filter(n => {
  const p = n.filePath.toLowerCase();
  return p.includes('docker') || p.includes('.tf') || p.endsWith('makefile');
}).map(n => n.filePath);
const deploymentTopology = {
  hasDockerfile: infraFiles.some(f => f.toLowerCase().includes('dockerfile')),
  hasCompose: infraFiles.some(f => f.includes('docker-compose')),
  hasK8s: false, hasTerraform: false, hasCI: false, infraFiles
};

// I. Data pipeline
const dataPipeline = {
  schemaFiles: fileNodes.filter(n => n.filePath.match(/\.(sql|graphql|gql|proto)$/)).map(n=>n.filePath),
  migrationFiles: fileNodes.filter(n => n.filePath.includes('migration')).map(n=>n.filePath),
  dataModelFiles: fileNodes.filter(n => n.tags && n.tags.includes('data')).map(n=>n.filePath),
  apiHandlerFiles: fileNodes.filter(n => n.tags && (n.tags.includes('api-handler')||n.tags.includes('inference'))).map(n=>n.filePath)
};

// J. Doc coverage
const docGroups = new Set();
for (const n of fileNodes) if (n.type === 'document') { const grp = nodeToGroup[n.id]; if(grp) docGroups.add(grp); }
const allGroups = Object.keys(directoryGroups);
const docCoverage = {
  groupsWithDocs: docGroups.size,
  totalGroups: allGroups.length,
  coverageRatio: allGroups.length ? docGroups.size/allGroups.length : 0,
  undocumentedGroups: allGroups.filter(g => !docGroups.has(g))
};

// K. Dependency direction
const dependencyDirection = interGroupImports.map(({from,to}) => ({dependent:from, dependsOn:to}));

const fileStats = {
  totalFileNodes: fileNodes.length,
  filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([g,ids])=>[g,ids.length])),
  nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([t,ids])=>[t,ids.length]))
};

const result = {
  scriptCompleted: true,
  directoryGroups, nodeTypeGroups, crossCategoryEdges,
  interGroupImports, intraGroupDensity, patternMatches,
  deploymentTopology, dataPipeline, docCoverage, dependencyDirection,
  fileStats,
  fileFanIn: Object.fromEntries(Object.entries(fanIn).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1])),
  fileFanOut: Object.fromEntries(Object.entries(fanOut).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]))
};

fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
console.log('Done. Nodes:', fileNodes.length);
