export const workbenchViews = ['review', 'evidence', 'behavior', 'knowledge', 'report', 'teaching'] as const
export type WorkbenchView = typeof workbenchViews[number]
export type BehaviorView = 'overview' | 'heatmap' | 'frames' | 'models'
export type EvidenceView = 'content' | 'courseware'

export function parseWorkbenchRoute(search: string): { view: WorkbenchView; lesson: string | null } {
  const query = new URLSearchParams(search)
  const candidate = query.get('view')
  return { view: workbenchViews.includes(candidate as WorkbenchView) ? candidate as WorkbenchView : 'review', lesson: query.get('lesson') }
}

// Preserve modern query parameters and retire the removed snapshot entry points.
export function workbenchUrl(href: string, view: WorkbenchView, lesson?: string): URL {
  const url = new URL(href)
  for (const flag of ['snapshots', 'data', 'demo']) url.searchParams.delete(flag)
  url.searchParams.set('view', view)
  if (lesson) url.searchParams.set('lesson', lesson)
  url.hash = ''
  return url
}
