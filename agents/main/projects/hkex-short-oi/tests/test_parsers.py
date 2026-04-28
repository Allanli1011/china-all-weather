import unittest
from datetime import date

from hkex_short_oi.parsers import (
    ReportUnavailable,
    parse_daily_quote_short_turnover,
    parse_short_turnover_page,
)
from hkex_short_oi.sfc import find_sfc_csv_links, parse_sfc_positions_csv


SHORT_PAGE_SAMPLE = """
Short Selling Turnover (Main Board) up to morning close today
              SHORTSELL REPORT   TRADING DATE : 28 APR 2026 (TUESDAY)

                                              Turnover
      CODE   NAME OF STOCK               (SH)            ($)
          5  HSBC HOLDINGS            849,200    118,913,440
        700  TENCENT                1,118,900    533,560,040
      %80016  SHK PPT-R                  2,500        300,000
      % 9008  BOS HSK BTC-U                500            379
    Total No. of all Securities recording Short Selling                        :   799
    (A) Short Selling of Designated Securities (excluding ETP)
        Short Selling Turnover Total Shares (SH)                  :        764,782,240
        Short Selling Turnover Total Value ($)                    : HKD 17,083,178,864
    Total market turnover                                         : HKD131,190,943,689
    Short Selling of Designated Securities (excluding ETP) as % total turnover :   13%
    (C) Short Selling of all Designated Securities
        Short Selling Turnover Total Shares (SH)                  :        764,782,240
        Short Selling Turnover Total Value ($)                    : HKD 17,083,178,864
    Total market turnover                                         : HKD131,190,943,689
    Short Selling of all Designated Securities as % total turnover             :   13%
    *Total No. of non-Designated Securities recording Short Selling            :     0
        Short Selling Turnover Total Shares (SH)                  :                  0
        Short Selling Turnover Total Value ($)                    : HKD              0
"""


DAILY_QUOTE_SAMPLE = """
                           THE STOCK EXCHANGE OF HONG KONG LIMITED
    EXCHANGE SQUARE, HONG KONG      TEL: 25221122      DATE: 09 APR 2025 (WEDNESDAY)

                          SHORT SELLING TURNOVER - DAILY REPORT

                                 Total Short Selling Turnover             Total Turnover
      CODE  NAME OF STOCK            (SH)             ($)             (SH)                ($)

       8083 YOUZAN               12,444,000           1,035,288     114,578,118           9,512,252
       8137 HONBRIDGE                70,000              40,400         944,000             543,960

       Total Shares (SH):        12,514,000                         115,522,118
                       PREVIOUS DAY'S ADJUSTED SHORT SELLING TURNOVER
"""


class ParserTests(unittest.TestCase):
    def test_parse_short_turnover_page_handles_codes_and_summaries(self):
        report = parse_short_turnover_page(
            SHORT_PAGE_SAMPLE,
            market="main",
            session="morning",
            source_url="https://example.test/mshtmain.htm",
        )

        self.assertEqual(report.trade_date, date(2026, 4, 28))
        self.assertEqual(len(report.records), 4)
        self.assertEqual(report.records[0].code, "00005")
        self.assertEqual(report.records[1].code, "00700")
        self.assertEqual(report.records[2].code, "80016")
        self.assertTrue(report.records[2].is_non_hkd)
        self.assertEqual(report.records[3].code, "09008")
        self.assertTrue(report.records[3].is_non_hkd)
        self.assertEqual(report.records[1].short_value, 533_560_040)
        self.assertEqual(report.summaries[0].short_values["HKD"], 17_083_178_864)
        self.assertEqual(report.summaries[0].short_pct_market, 13.0)
        self.assertEqual(report.summaries[-1].short_values["HKD"], 17_083_178_864)

    def test_parse_short_turnover_page_unavailable(self):
        with self.assertRaises(ReportUnavailable):
            parse_short_turnover_page(
                "Short Selling Turnover (Main Board) up to day close today will be available after day close.",
                market="main",
                session="day",
                source_url="https://example.test/ashtmain.htm",
            )

    def test_parse_daily_quote_short_turnover_includes_total_turnover(self):
        report = parse_daily_quote_short_turnover(
            DAILY_QUOTE_SAMPLE,
            market="gem",
            source_url="https://example.test/e_G250409.htm",
        )

        self.assertEqual(report.trade_date, date(2025, 4, 9))
        self.assertEqual(len(report.records), 2)
        youzan = report.records[0]
        self.assertEqual(youzan.code, "08083")
        self.assertEqual(youzan.total_value, 9_512_252)
        self.assertAlmostEqual(youzan.short_ratio, 1_035_288 / 9_512_252)

    def test_parse_sfc_positions_csv(self):
        text = "\n".join(
            [
                "Date,Stock Code,Stock Name,Aggregated Reportable Short Positions (Shares),Aggregated Reportable Short Positions (HK$)",
                "17/04/2026,5,HSBC HOLDINGS,64312593,9100231910",
            ]
        )
        records = parse_sfc_positions_csv(text, source_url="https://example.test/sfc.csv")

        self.assertEqual(records[0].report_date, date(2026, 4, 17))
        self.assertEqual(records[0].code, "00005")
        self.assertEqual(records[0].short_position_hkd, 9_100_231_910)

    def test_find_sfc_csv_links(self):
        html = (
            '<a href="https://www.sfc.hk/-/media/EN/pdf/spr/2026/04/17/'
            'Short_Position_Reporting_Aggregated_Data_20260417.csv?rev=1">CSV</a>'
        )
        self.assertEqual(
            find_sfc_csv_links(html),
            [
                "https://www.sfc.hk/-/media/EN/pdf/spr/2026/04/17/"
                "Short_Position_Reporting_Aggregated_Data_20260417.csv?rev=1"
            ],
        )


if __name__ == "__main__":
    unittest.main()
