"""Benchmark SKNN-QP seguro: 4 datasets, k específico, y salida Excel.

Dependencias: numpy, openpyxl y módulos del proyecto CloudB, KeyM, SecureDataT.
Uso: python ejecutar_benchmark_seguro.py
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from pathlib import Path

CONFIGURACIONES = ((10, 3), (30, 5), (50, 7), (100, 10))


def leer_csv(ruta):
    with ruta.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def convertir(valor):
    if valor is None or valor == '':
        return ''
    if valor in ('True', 'False'):
        return valor == 'True'
    try:
        if valor.strip().lstrip('+-').isdigit():
            return int(valor)
        return float(valor)
    except (ValueError, AttributeError):
        return valor


def crear_excel(resultados, destino, configuraciones=CONFIGURACIONES):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.dimensions import ColumnDimension

    libro = Workbook()
    resumen = libro.active
    resumen.title = 'Resumen'
    resumen.merge_cells('A1:O1')
    resumen['A1'] = 'Benchmark SKNN-QP · Paillier · Letter Recognition'
    resumen['A1'].fill = PatternFill('solid', fgColor='17324D')
    resumen['A1'].font = Font(name='Aptos', bold=True, color='FFFFFF', size=15)
    resumen.row_dimensions[1].height = 34
    resumen.merge_cells('A2:O2')
    resumen['A2'] = 'Tiempos en segundos. Solo se muestran mediciones reales de ejecuciones completadas.'
    resumen['A2'].font = Font(name='Aptos', italic=True, color='53657A')
    columnas = ['Registros', 'k', 'Consultas', 'Atributos', 'Bits clave', 'Altura máx.',
                'Outsourcing (s)', 'Consulta cifrada prom. (s)', 'Descifrado prom. (s)',
                'Consulta total prom. (s)', 'Tiempo total (s)', 'Exactitud', 'F1 macro',
                'Top-k correcto', 'Distancias correctas']
    resumen.append([])
    resumen.append(columnas)
    for cell in resumen[4]:
        cell.fill = PatternFill('solid', fgColor='176B91')
        cell.font = Font(bold=True, color='FFFFFF')
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    resumen.row_dimensions[4].height = 32
    for idx, (n,k) in enumerate(configuraciones, start=5):
        r = resultados.get(n,{})
        valores = [n,k, convertir(r.get('consultas','')), convertir(r.get('atributos_m','')),
                   convertir(r.get('bits_paillier','')), convertir(r.get('altura_maxima_h','')),
                   convertir(r.get('tiempo_outsourcing_s','')), convertir(r.get('tiempo_consulta_promedio_s','')),
                   convertir(r.get('tiempo_descifrado_promedio_s','')),
                   convertir(r.get('tiempo_consulta_total_promedio_s','')),
                   convertir(r.get('tiempo_total_s','')), convertir(r.get('accuracy','')),
                   convertir(r.get('f1_macro','')), convertir(r.get('topk_correcto_todas','')),
                   convertir(r.get('distancias_coinciden_todas',''))]
        resumen.append(valores)
        for cell in resumen[idx]:
            if idx % 2 == 0:
                cell.fill = PatternFill('solid', fgColor='EDF4FA')
            if 7 <= cell.column <= 11 and isinstance(cell.value, (float,int)):
                cell.number_format = '0.000000'
            if cell.column in (12,13) and isinstance(cell.value, (float,int)):
                cell.number_format = '0.00%'
    for c in range(1,16):
        resumen.column_dimensions[get_column_letter(c)].width = (14 if c==1 else 23 if 7<=c<=11 else 21 if c>=12 else 14)
    resumen.freeze_panes = 'A5'
    resumen.auto_filter.ref = f'A4:O{4+len(configuraciones)}'

    for n,k in configuraciones:
        hoja = libro.create_sheet(f'Consultas_{n}')
        archivo = destino.parent / f'consultas_seguro_n{n}.csv'
        if archivo.exists() and n in resultados:
            filas = leer_csv(archivo)
            if filas:
                llaves = list(filas[0])
                hoja.append(llaves)
                for fila in filas:
                    hoja.append([convertir(fila.get(c,'')) for c in llaves])
                for celda in hoja[1]:
                    celda.fill = PatternFill('solid',fgColor='176B91')
                    celda.font = Font(bold=True,color='FFFFFF')
                    celda.alignment = Alignment(wrap_text=True)
                hoja.row_dimensions[1].height = 32
                for c in range(1,len(llaves)+1):
                    hoja.column_dimensions[get_column_letter(c)].width = 22
                hoja.freeze_panes = 'A2'
                hoja.auto_filter.ref = hoja.dimensions
                continue
        hoja.append(['Configuración',f'BD={n}, k={k}'])
        hoja.append(['Estado','Pendiente de ejecutar SKNN-QP'])
        for c in hoja[1]:
            c.fill = PatternFill('solid',fgColor='176B91')
            c.font = Font(bold=True,color='FFFFFF')
        hoja.column_dimensions['A'].width = 28
        hoja.column_dimensions['B'].width = 42
    destino.parent.mkdir(parents=True, exist_ok=True)
    libro.save(destino)
    print(f'Excel generado: {destino.resolve()}')


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description='Ejecuta n=10/k=3, n=30/k=5, n=50/k=7 y n=100/k=10')
    p.add_argument('--preprocessed-dir', type=Path, default=base / 'salida_preprocesamiento')
    p.add_argument('--output-dir', type=Path, default=base / 'resultados_seguro')
    p.add_argument('--queries', type=int, default=10)
    p.add_argument('--key-bits', type=int, default=1024)
    p.add_argument('--attributes', type=int, default=16)
    p.add_argument('--max-depth', type=int, default=7)
    p.add_argument('--fixed-point-bits', type=int, default=12)
    p.add_argument('--leaf-capacity', type=int, default=1)
    p.add_argument('--solo-excel', action='store_true', help='Consolida CSV existentes, sin ejecutar cifrado')
    args = p.parse_args()
    if not 1 <= args.queries <= 10:
        p.error('El benchmark contiene 10 consultas por dataset')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resultados = {}
    for n,k in CONFIGURACIONES:
        csv_res = args.output_dir / f'resumen_seguro_n{n}.csv'
        if not args.solo_excel:
            print(f'\nEjecutando SKNN-QP seguro: BD={n}, k={k}, consultas={args.queries}', flush=True)
            cmd = [sys.executable, str(base / 'experimento4.py'),
                   '--preprocessed-dir',str(args.preprocessed_dir), '--output-dir',str(args.output_dir),
                   '--db-rows',str(n), '--k',str(k), '--queries',str(args.queries),
                   '--attributes',str(args.attributes), '--key-bits',str(args.key_bits),
                   '--max-depth',str(args.max_depth), '--fixed-point-bits',str(args.fixed_point_bits),
                   '--leaf-capacity',str(args.leaf_capacity)]
            subprocess.run(cmd, check=True, cwd=base)
        if csv_res.exists():
            filas = leer_csv(csv_res)
            if filas:
                if int(filas[0]['k']) != k:
                    raise ValueError(f'CSV {csv_res} tiene un k distinto de {k}')
                resultados[n] = filas[0]
    crear_excel(resultados, args.output_dir / 'benchmark_seguro_resultados.xlsx')


main()
