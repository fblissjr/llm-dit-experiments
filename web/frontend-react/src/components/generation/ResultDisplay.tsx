/**
 * Result Display
 *
 * Shows the current generation result with generous whitespace.
 * Results are hero-sized content, not decoration.
 */

import { useGenerationStore } from '@/stores/generationStore';
import { usePipelineStore } from '@/stores/pipelineStore';
import { ProgressDisplay } from './ProgressDisplay';
import { formatDuration } from '@/types/generation';

export function ResultDisplay() {
  const { status, currentResult, error } = useGenerationStore();
  const { selectedPipelineId, pipelines } = usePipelineStore();
  const pipeline = selectedPipelineId ? pipelines[selectedPipelineId] : null;

  // Show progress if generating
  if (status === 'generating' || status === 'loading') {
    return (
      <div className="space-y-4">
        <ProgressDisplay />
        {status === 'loading' && (
          <div className="card text-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-400">Loading model...</p>
          </div>
        )}
      </div>
    );
  }

  // Show error
  if (status === 'error' && error) {
    return (
      <div className="card border-red-500/50 bg-red-900/20">
        <div className="flex items-start gap-3">
          <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <h3 className="font-medium text-red-400">Generation Failed</h3>
            <p className="text-sm text-gray-400 mt-1">{error.message}</p>
            {error.recoverable && (
              <p className="text-xs text-gray-500 mt-2">
                Try adjusting parameters or check model status.
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Show result
  if (currentResult) {
    return (
      <div className="space-y-4">
        {/* Main result display */}
        <div className="result-container">
          {currentResult.outputType === 'video' ? (
            <video
              src={currentResult.urls[0]}
              controls
              loop
              autoPlay
              muted
              className="w-full rounded-xl"
            />
          ) : currentResult.outputType === 'layers' ? (
            // Layer output - show grid of images
            <div className="grid grid-cols-2 gap-2 p-2">
              {currentResult.urls.map((url, i) => (
                <img
                  key={i}
                  src={url}
                  alt={`Layer ${i + 1}`}
                  className="w-full rounded-lg"
                />
              ))}
            </div>
          ) : (
            // Single image
            <img
              src={currentResult.urls[0]}
              alt="Generated result"
              className="w-full rounded-xl"
            />
          )}
        </div>

        {/* Result metadata */}
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center gap-4">
            <span>Seed: {currentResult.seed}</span>
            <span>Time: {formatDuration(currentResult.durationMs)}</span>
          </div>
          <div className="flex items-center gap-2">
            <DownloadButton urls={currentResult.urls} outputType={currentResult.outputType} />
          </div>
        </div>
      </div>
    );
  }

  // Empty state
  return (
    <div className="card text-center py-16">
      <div className={`text-6xl mb-4 opacity-50`}>
        {pipeline?.icon ?? '🎨'}
      </div>
      <h3 className="text-lg font-medium text-gray-300 mb-2">
        Ready to Generate
      </h3>
      <p className="text-sm text-gray-500 max-w-sm mx-auto">
        Configure your settings and click Generate to create{' '}
        {pipeline?.output_type === 'video' ? 'a video' : 'an image'}.
      </p>
    </div>
  );
}

interface DownloadButtonProps {
  urls: string[];
  outputType: string;
}

function DownloadButton({ urls, outputType }: DownloadButtonProps) {
  const handleDownload = () => {
    urls.forEach((url, index) => {
      const link = document.createElement('a');
      link.href = url;
      const ext = outputType === 'video' ? 'mp4' : 'png';
      link.download = urls.length > 1 ? `output-${index + 1}.${ext}` : `output.${ext}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  };

  return (
    <button
      onClick={handleDownload}
      className="btn-ghost text-sm py-1.5 px-3"
    >
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
      Download
    </button>
  );
}
