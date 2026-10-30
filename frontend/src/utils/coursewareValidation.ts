// Courseware has no inherent classroom clock. Only observed matches supply time.
export function validateCourseware(data: any, media: (url: unknown) => string) {
  const fail = (message: string): never => { throw new Error(`Invalid courseware evidence: ${message}`) }
  const array = (x: any, field: string): any[] => Array.isArray(x) ? x : fail(field)
  const text = (x: any, field: string) => typeof x === 'string' && x.trim() ? x : fail(field)
  const timed = (a: any, b: any) => Math.abs(a.start_sec - b.start_sec) <= .01 && Math.abs(a.end_sec - b.end_sec) <= .01
  if (!data.courseware) {
    if (['courseware_evidence', 'transcript_revisions', 'revised_evidence_blocks'].some(key => data[key]?.length)) fail('Additional evidence has no corresponding courseware')
    return
  }
  const cw = data.courseware
  for (const key of ['document_id', 'filename', 'sha256', 'format']) text(cw[key], key)
  media(cw.source_url)
  const pages = new Map<string, any>(), matches = new Map<string, any>(), transcripts = new Map<string, any>(data.transcript.map((r: any) => [r.id, r]))
  const seen = new Set<string>([...data.transcript, ...data.slides, ...data.alignment, ...(data.evidence_blocks || [])].map((r: any) => r.id))
  const add = (row: any) => { text(row.id, 'id'); if (seen.has(row.id)) fail(`Duplicate ID ${row.id}`); seen.add(row.id) }
  for (const page of array(cw.pages, 'pages')) {
    add(page); if (!Number.isInteger(page.page_number) || page.page_number < 1 || typeof page.text !== 'string') fail('Page number or source text')
    media(page.image_url); pages.set(page.id, page)
  }
  for (const match of array(cw.matches, 'matches')) {
    add(match)
    const slide = data.slides.find((r: any) => r.id === match.slide_id)
    if (!slide || !timed(match, slide)) fail('Match times must come from original screen observations')
    if (!['auto_matched', 'confirmed', 'ambiguous', 'unmatched', 'rejected'].includes(match.status)) fail('Match status')
    if (match.page_id !== null && !pages.has(match.page_id)) fail('Matched page does not belong to the current courseware')
    if (['auto_matched', 'confirmed'].includes(match.status) && !pages.has(match.page_id)) fail('Valid match is missing a page')
    matches.set(match.id, match)
  }
  for (const row of array(data.courseware_evidence, 'courseware_evidence')) {
    add(row)
    const ids = row.match_ids || (row.match_id ? [row.match_id] : [])
    if (!ids.length || row.kind !== 'courseware' || !pages.has(row.page_id) || row.text !== pages.get(row.page_id).text) fail('Courseware citation text or page')
    for (const id of ids) {
      const match = matches.get(id)
      if (!match || !['auto_matched', 'confirmed'].includes(match.status) || match.page_id !== row.page_id || !timed(row, match)) fail('Courseware citation is unmatched or has invalid times')
    }
  }
  const revisions = new Map<string, any>()
  for (const row of array(data.transcript_revisions, 'transcript_revisions')) {
    add(row)
    const original = transcripts.get(row.transcript_id)
    if (!original || row.original_text !== original.text || !timed(row, original) || row.status !== 'suggested') fail('Revision does not match the original transcript')
    let rebuilt = '', cursor = 0
    for (const edit of array(row.edits, 'edits')) {
      if (!Number.isInteger(edit.start_char) || !Number.isInteger(edit.end_char) || edit.start_char < cursor || edit.end_char <= edit.start_char || original.text.slice(edit.start_char, edit.end_char) !== edit.original) fail('Revision range')
      if (!array(edit.page_ids, 'page_ids').length || !array(edit.match_ids, 'match_ids').length) fail('Revision is missing sources')
      for (const id of edit.page_ids) if (!pages.has(id) || !pages.get(id).text.includes(edit.replacement)) fail('Term does not appear in the source courseware')
      for (const id of edit.match_ids) {
        const match = matches.get(id)
        if (!match || !['auto_matched', 'confirmed'].includes(match.status) || !edit.page_ids.includes(match.page_id) || match.end_sec <= row.start_sec || match.start_sec >= row.end_sec) fail('Term source is not aligned')
      }
      rebuilt += original.text.slice(cursor, edit.start_char) + edit.replacement; cursor = edit.end_char
    }
    rebuilt += original.text.slice(cursor)
    if (rebuilt !== row.corrected_text) fail('Revision text does not match its record')
    revisions.set(row.id, row)
  }
  for (const row of array(data.revised_evidence_blocks, 'revised_evidence_blocks')) {
    const refs = array(row.source_refs, 'source_refs'), sources = refs.map(id => transcripts.get(id))
    if (!sources.length || sources.some(r => !r)) fail('Revised retrieval block original sources')
    const span = { start_sec: Math.min(...sources.map(r => r.start_sec)), end_sec: Math.max(...sources.map(r => r.end_sec)) }
    if (!timed(row, span)) fail('Revised retrieval block times')
    const revisionIds = row.revision_ids || []
    if (revisionIds.some((id: string) => !revisions.has(id) || !refs.includes(revisions.get(id).transcript_id))) fail('Revision references')
    const revised = new Map(revisionIds.map((id: string) => [revisions.get(id).transcript_id, revisions.get(id).corrected_text]))
    if (row.text !== sources.map(r => revised.get(r.id) ?? r.text).join(' ')) fail('Revised retrieval block text')
    if (row.kind === 'transcript_revision') { add(row); if (!revisionIds.length) fail('Revised block has no revisions') }
    else if (row.kind === 'transcript') {
      const raw = [...(data.evidence_blocks || []), ...data.transcript].find((r: any) => r.id === row.id)
      if (!raw || raw.text !== row.text || !timed(raw, row) || revisionIds.length) fail('Original retrieval block has changed')
    } else fail('Revision block kind')
  }
}
