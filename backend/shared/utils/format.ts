export function formatNumber(value: number, decimals: number = 2): string {
  if (value === undefined || value === null || isNaN(value)) return 'N/A';
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

export function formatPercent(value: number): string {
  if (value === undefined || value === null || isNaN(value)) return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
}

export function formatCurrency(value: number): string {
  if (value === undefined || value === null || isNaN(value)) return 'N/A';
  return value.toLocaleString(undefined, {
    style: 'currency',
    currency: 'USD'
  });
}
