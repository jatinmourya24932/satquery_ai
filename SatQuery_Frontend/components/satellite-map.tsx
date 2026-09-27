'use client';

import { useEffect, useRef } from 'react';
import {
  MapContainer,
  TileLayer,
  Polygon,
  Marker,
  Popup,
  useMap,
} from 'react-leaflet';
import L from 'leaflet';

delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const KHANDWA_CENTER: [number, number] = [21.8264, 76.3548];

const KHANDWA_POLYGON: [number, number][] = [
  [21.95, 76.15],
  [21.95, 76.55],
  [21.65, 76.55],
  [21.65, 76.15],
];

interface SatelliteMapProps {
  showPolygon: boolean;
  layerType: string;
}

function MapResizer() {
  const map = useMap();
  useEffect(() => {
    const handler = () => map.invalidateSize();
    handler();
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, [map]);
  return null;
}

const layerFilters: Record<string, string> = {
  'True Color': '',
  'NDVI Vegetation':
    'invert(1) hue-rotate(90deg) saturate(1.5) contrast(1.2) brightness(0.85)',
  LULC: 'hue-rotate(30deg) saturate(1.8) contrast(1.3) brightness(0.9)',
  'Water Mask': 'invert(1) hue-rotate(200deg) saturate(2) contrast(1.5)',
  Urban: 'grayscale(0.3) sepia(0.4) hue-rotate(10deg) contrast(1.2)',
  'Change Detection': 'hue-rotate(280deg) saturate(1.6) contrast(1.3)',
};

export function SatelliteMap({ showPolygon, layerType }: SatelliteMapProps) {
  const mapRef = useRef<L.Map | null>(null);

  return (
    <MapContainer
      center={KHANDWA_CENTER}
      zoom={11}
      scrollWheelZoom
      className="h-full w-full"
      zoomControl={true}
      ref={(map) => {
        if (map) mapRef.current = map;
      }}
    >
      <MapResizer />
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attribution='Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics'
        maxZoom={19}
        className={layerType !== 'True Color' ? 'satellite-filter' : ''}
      />
      <style>{`
        .satellite-filter .leaflet-tile-pane {
          filter: ${layerFilters[layerType] || ''};
        }
      `}</style>

      <Marker position={KHANDWA_CENTER}>
        <Popup>
          <div className="text-xs">
            <strong>Khandwa, MP</strong>
            <br />
            Center of Analysis Region
          </div>
        </Popup>
      </Marker>

      {showPolygon && (
        <Polygon
          positions={KHANDWA_POLYGON}
          pathOptions={{
            color: '#DC2626',
            weight: 3,
            fillColor: '#F97316',
            fillOpacity: 0.2,
            dashArray: '10 6',
            className: 'animate-draw-polygon',
          }}
        />
      )}
    </MapContainer>
  );
}
