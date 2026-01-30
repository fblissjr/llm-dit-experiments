/**
 * Notifications
 *
 * Toast-style notifications displayed in corner.
 */

import { useUIStore } from '@/stores/uiStore';

export function Notifications() {
  const { notifications, removeNotification } = useUIStore();

  if (notifications.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2 max-w-sm">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={`
            p-4 rounded-lg shadow-lg border backdrop-blur-sm
            animate-slide-up
            ${getNotificationStyles(notification.type)}
          `}
        >
          <div className="flex items-start gap-3">
            <NotificationIcon type={notification.type} />
            <div className="flex-1 min-w-0">
              <p className="text-sm">{notification.message}</p>
            </div>
            <button
              onClick={() => removeNotification(notification.id)}
              className="p-1 text-gray-400 hover:text-gray-200 transition-colors flex-shrink-0"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function getNotificationStyles(type: string): string {
  switch (type) {
    case 'success':
      return 'bg-green-900/80 border-green-700 text-green-100';
    case 'error':
      return 'bg-red-900/80 border-red-700 text-red-100';
    case 'warning':
      return 'bg-yellow-900/80 border-yellow-700 text-yellow-100';
    default:
      return 'bg-gray-800/80 border-gray-700 text-gray-100';
  }
}

function NotificationIcon({ type }: { type: string }) {
  const iconClass = 'w-5 h-5 flex-shrink-0';

  switch (type) {
    case 'success':
      return (
        <svg className={`${iconClass} text-green-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      );
    case 'error':
      return (
        <svg className={`${iconClass} text-red-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      );
    case 'warning':
      return (
        <svg className={`${iconClass} text-yellow-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    default:
      return (
        <svg className={`${iconClass} text-blue-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
  }
}
