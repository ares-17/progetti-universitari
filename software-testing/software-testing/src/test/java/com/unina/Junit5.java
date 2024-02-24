package com.unina;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.platform.commons.util.StringUtils;

class Junit5 {

	@BeforeAll
	static void beforeAll() {
		/*
		 * assumeTrue è utilizzato per verificare se una determinata condizione. Se la
		 * codizione fallisce, il test/ i test seguenti sono ignorati e non eseguiti.
		 * Con assertTrue invece i test sono sempre eseguiti.
		 */
		assumeTrue(true);
		System.out.print("before all");
	}

	@BeforeEach
	void beforeEach() {
		System.out.print("before each");
	}

	@AfterEach
	void tearDown() {
		System.out.print("after each");
	}

	@Test
	@DisplayName("a stupid test")
	void test() {
		assertTrue(true);
	}

	@ParameterizedTest
	@ValueSource(strings = { "racecar", "radar", "able was I ere I saw elba" })
	// Esegue la funzione di test su tutti gli elementi di input
	void parameterizedTest(String candidate) {
		assertTrue(StringUtils.isNotBlank(candidate));
	}

	@ParameterizedTest
	@MethodSource("provideNumeratorsAndDenominators")
	/*
	 * In Junit5 @ParameterizedTest sostituisce @Theory accompagnato da @DataPoints
	 */
	public void naturali(int n, int d) {
		assumeTrue(n > 0 && d > 0);
		System.out.println("Test of " + n + " / " + d);
		double res = (double) n / d;
		assertTrue(res > 0);
	}

	static Stream<Integer[]> provideNumeratorsAndDenominators() {
		return Stream.of(new Integer[] { 100, 4 }, new Integer[] { 12, 9 }, new Integer[] { 17, 12 },
				new Integer[] { 3, 123 }, new Integer[] { 15, 88 }, new Integer[] { 88, 1 });
	}

	@RepeatedTest(2)
	// utile quando il risultato non dipende esclusivamente dagli input della
	// funzione
	void repeatedTest() {
		assertDoesNotThrow(() -> {
			return 2 / 2;
		});
	}

	@TestFactory
	/*
	 * Con @TestFactory le funzioni @BeforeEach e @AfterEach non sono invocate.
	 * Genera in maniera del tutto dinamica i test.
	 * Particolarmente utile per test che dipendono da altri fattori
	 */
	Collection<DynamicTest> dynamicTestsFromCollection() {
		return Arrays.asList(
			dynamicTest("1st dynamic test", () -> assertTrue(StringUtils.isNotBlank("is not blank"))),
			dynamicTest("2nd dynamic test", () -> assertEquals(4, 2*2)));
	}

}
