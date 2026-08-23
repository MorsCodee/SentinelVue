function CornerBrackets({ color = "ink" }) {
  const colorClass = `stroke-${color}`;

  return (
    <>
      {/* Top-left bracket */}
      <svg className={`absolute top-3 left-3 w-6 h-6 ${colorClass}`} viewBox="0 0 24 24" fill="none">
        <path d="M2 8V2H8" strokeWidth="2" />
      </svg>
      {/* Top-right bracket */}
      <svg className={`absolute top-3 right-3 w-6 h-6 ${colorClass}`} viewBox="0 0 24 24" fill="none">
        <path d="M16 2H22V8" strokeWidth="2" />
      </svg>
      {/* Bottom-left bracket */}
      <svg className={`absolute bottom-3 left-3 w-6 h-6 ${colorClass}`} viewBox="0 0 24 24" fill="none">
        <path d="M2 16V22H8" strokeWidth="2" />
      </svg>
      {/* Bottom-right bracket */}
      <svg className={`absolute bottom-3 right-3 w-6 h-6 ${colorClass}`} viewBox="0 0 24 24" fill="none">
        <path d="M16 22H22V16" strokeWidth="2" />
      </svg>
    </>
  );
}

export default CornerBrackets;