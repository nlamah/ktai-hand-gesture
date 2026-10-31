import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';
const dir = new URL('.', import.meta.url).pathname;
const p = JSON.parse(await fs.readFile(dir+'proposal_analysis.json','utf8'));
const wb = Workbook.create();
const features=wb.worksheets.add('Features');
const results=wb.worksheets.add('Resultaten');
const changes=wb.worksheets.add('Wijzigingen');
const dark='#263F56', pale='#FFF1BF';
function base(s,range){s.showGridLines=false;s.getRange(range).format.font={name:'Arial',size:10};s.getRange(range).format.verticalAlignment='center';s.getRange(range).format.rowHeight=26;}
function header(s,range){s.getRange(range).format.fill=dark;s.getRange(range).format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};s.getRange(range).format.wrapText=true;s.getRange(range).format.rowHeight=42;}
base(features,'A1:T27');features.getRange('A1:T27').values=p.rows;
features.getRange('A1:A27').format.columnWidth=9;
features.getRange('B1:T27').format.columnWidth=22;
features.getRange('B2:T27').format.wrapText=true;
features.getRange('A2:T27').format.rowHeight=36;
header(features,'A1:T1');features.freezePanes.freezeRows(1);features.freezePanes.freezeColumns(1);
const featureKeys=['thumb_mcp_shape','flexion_thumb_ip','flexion_index_mcp','flexion_index_pip','flexion_index_dip','flexion_middle_mcp','flexion_middle_pip','flexion_middle_dip','flexion_ring_mcp','flexion_ring_pip','flexion_ring_dip','flexion_pinky_mcp','flexion_pinky_pip','flexion_pinky_dip','spread_index_middle','spread_middle_ring','crossing_index_middle','thumb_tip_index_proximity','thumb_tip_middle_proximity'];
for(const c of p.changes){features.getCell(c.letter.charCodeAt(0)-64,featureKeys.indexOf(c.feature)+1).format.fill=pale;}
base(results,'A1:H57');results.getRange('A1:A57').format.columnWidth=44;results.getRange('B1:H57').format.columnWidth=18;
results.getRange('A2').values=[['Voorstel ruimere featurewaarden']];results.getRange('A2').format.font={name:'Arial',size:14,bold:true};
results.getRange('A4:D4').values=[['Trainingsdata: alle beelden','Aantal beelden','Juiste letter volledig','Aandeel']];header(results,'A4:D4');
results.getRange('A5:C6').values=[['Huidige tabel',p.comparison.all.before.images,p.comparison.all.before.correct_letter_full_matches],['Voorstel',p.comparison.all.proposal.images,p.comparison.all.proposal.correct_letter_full_matches]];
results.getRange('D5:D6').formulas=[['=C5/B5'],['=C6/B6']];results.getRange('D5:D6').setNumberFormat('0.0%');
results.getRange('A8:D8').values=[['Controle / ontwikkeling','Aantal beelden','Juiste letter volledig','Aandeel']];header(results,'A8:D8');
results.getRange('A9:C10').values=[['Huidige tabel',303,p.comparison.holdout.before.correct_letter_full_matches],['Voorstel',303,p.comparison.holdout.proposal.correct_letter_full_matches]];results.getRange('D9:D10').formulas=[['=C9/B9'],['=C10/B10']];results.getRange('D9:D10').setNumberFormat('0.0%');
results.getRange('A12:C16').values=[['Overige maten','Huidig','Voorstel'],['Featurewaarden voldaan',p.comparison.all.before.feature_match_fraction,p.comparison.all.proposal.feature_match_fraction],['Uniek juiste volledige matches',p.comparison.all.before.unique_correct_full_matches,p.comparison.all.proposal.unique_correct_full_matches],['Beelden met verkeerde lettermatch',p.comparison.all.before.images_with_any_wrong_full_match,p.comparison.all.proposal.images_with_any_wrong_full_match],['Verkeerde beeld-lettercombinaties',p.comparison.all.before.wrong_letter_full_acceptances,p.comparison.all.proposal.wrong_letter_full_acceptances]];header(results,'A12:C12');results.getRange('B13:C13').setNumberFormat('0.0%');
results.getRange('A18:G18').values=[['Letter','Beelden','Huidig volledig','Voorstel volledig','Huidig aandeel','Voorstel aandeel','Gewijzigde cellen']];header(results,'A18:G18');
for(let i=0;i<26;i++){const r=i+19,x=p.per_letter[i];results.getRange(`A${r}:D${r}`).values=[[x.letter,x.before.images,x.before.correct_letter_full_matches,x.proposal.correct_letter_full_matches]];results.getRange(`E${r}:F${r}`).formulas=[[`=C${r}/B${r}`,`=D${r}/B${r}`]];results.getRange(`G${r}`).values=[[p.changes.filter(c=>c.letter===x.letter).length]];}
results.getRange('E19:F44').setNumberFormat('0.0%');
const notes=[
'Voorstel: maximaal 3 cellen per letter, telkens één aangrenzende buigstand erbij.',
'Geel op Features markeert wijzigingen. Slash = alleen de genoemde categorieën.',
'MCP, duim B/niet-B, spreiding, kruising en contact blijven behouden.',
'Juiste letter volledig = alle beoordeelde features van de juiste letter voldoen.',
'Dit is feature-overeenkomst, geen gemeten nauwkeurigheid van een letterclassifier.',
'315 beelden hebben geen bruikbare detectie; onbekend telt als niet voldaan.',
'Keuze op afstelgroepen; meetgrenzen vast. Geen testbeelden gebruikt.',
'Ontwikkelholdout is eerder gebruikt; geen onafhankelijk definitief testresultaat.',
'Snapshot: percentages reageren niet op handmatige wijzigingen op Features.',
'Bron: assets/asl_feature_references.xlsx; data/external/asl_letters_v1/train.',
];
for(let i=0;i<notes.length;i++)results.getRange(`A${47+i}`).values=[[notes[i]]];
base(changes,`A1:H${p.changes.length+1}`);changes.getRange('A1:H1').values=[['Letter','Feature','Huidige waarde','Voorgestelde waarde','Extra featurematches fit','Extra volledige juiste fit','Extra volledige verkeerde fit','Motivatie']];header(changes,'A1:H1');
changes.getRange('A2').write(p.changes.map(c=>[c.letter,p.rows[0][featureKeys.indexOf(c.feature)+1],c.before,c.after,c.fit_added_feature_matches,c.fit_own_images_added,c.fit_wrong_letter_acceptances_added,' Voorstel uit afstelbeelden; visueel beoordelen.']));
changes.getRange('A1:A50').format.columnWidth=8;changes.getRange('B1:D50').format.columnWidth=28;changes.getRange('E1:G50').format.columnWidth=19;changes.getRange('H1:H50').format.columnWidth=56;changes.getRange('B2:D50').format.wrapText=true;changes.getRange('A2:H50').format.rowHeight=36;changes.freezePanes.freezeRows(1);
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Resultaten!A4:D10',include:'values,formulas',tableMaxRows:10,tableMaxCols:4,maxChars:2000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!',options:{useRegex:true,maxResults:10},maxChars:500})).ndjson);
for(const [name,range] of [['Features','A1:G9'],['Features','H1:N9'],['Features','O1:T9'],['Resultaten','A1:G16'],['Wijzigingen','A1:H9']]){const blob=await wb.render({sheetName:name,range,scale:1.5,format:'png'});await fs.writeFile(dir+name+'_'+range.replace(':','_')+'.png',new Uint8Array(await blob.arrayBuffer()));}
await (await SpreadsheetFile.exportXlsx(wb)).save(dir+'ASL_featurevoorstel.xlsx');
